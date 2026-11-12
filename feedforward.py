import copy
import math
import os
import time

import numpy as np
from joblib import dump, load
from threadpoolctl import threadpool_limits

from activations import ACTIVATIONS, activation_fn, activation_backward
from data import apply_minmax, create_batches, minmax_stats
from losses import (
    check_loss, check_output_for_loss, clip_gradients, is_sigmoid_binary_crossentropy, is_softmax_crossentropy,
    loss_derivative_fn, loss_fn, sigmoid_binary_crossentropy_delta, softmax_crossentropy_delta,
)
from optimizers import make_optimizer
from progress import format_number, timing


# NumPy's matrix products run on a math library (BLAS) that splits them across all the CPU cores. The matrices of this
# project are small, and splitting them costs more than it saves: one MNIST epoch takes about 1.7 s with one thread
# and 2.8 s with 12. So it uses one thread. Call threadpool_limits again (e.g. threadpool_limits(4)) for larger models.
threadpool_limits(1)


# A dense (fully connected) layer: each neuron j computes a weighted sum of all the inputs plus its bias,
#   z[j] = sum over k of weights[j, k] * inputs[k] + biases[j]
# and passes it through the activation function: output[j] = f(z[j]).
#
# For a batch this is a single matrix product: (batch_size, num_inputs) x (num_inputs, num_neurons). A single sample
# with shape (num_inputs,) gives shape (num_neurons,).
def layer_forward(inputs, weights, biases, act_fn):
    z = np.dot(inputs, weights.T) + biases  # the biases are added to every sample of the batch
    return z, activation_fn(act_fn, z)


def load_model(filepath):
    model_params = load(filepath)
    layers = []

    for saved in model_params["layers"]:
        layer = {"weights": saved["weights"], "biases": saved["biases"], "act_fn": saved["act_fn"]}

        if "dropout" in saved:  # only in the layers that have it
            layer["dropout"] = saved["dropout"]

        layers.append(layer)

    return FeedForward(layers=layers, configs=model_params["configs"])


def clone_model(model):
    return FeedForward(layers=model.get_layers(), configs=model.get_configs())


# A feedforward neural network: a stack of dense layers, where the output of each layer is the input of the next.
#
# It learns with gradient descent (see train_step): it predicts, measures the loss, computes how the loss changes
# with every weight (backpropagation), and moves each weight a small step in the direction that lowers the loss.
class FeedForward:
    def __init__(self, layers=None, configs=None):
        # New containers for each model (a default argument of [] would be one list shared by all models)
        self.layers = layers if layers is not None else []     # weights, biases and activation of each layer
        self.configs = configs if configs is not None else {}  # loss, optimizer and input normalization
        self.optimizer = make_optimizer(self.configs.get("optimizer", "sgd"))  # applies the gradients (see update)
        self.cache = []    # inputs, z and output of each layer in the last forward pass, used by backward()
        self.history = {}  # loss of each epoch in the last train()
        self.observers = []  # objects told about each forward pass, update and metric (see monitor.py)

    # Tells the observers about an event: on_forward(model), on_update(model, gradients, learning_rate) or
    # on_metrics(model, values). Without observers it does nothing.
    def notify(self, event, *args):
        for observer in self.observers:
            getattr(observer, event)(self, *args)

    # dropout: the fraction of the inputs of this layer that the training sets to 0 at random, in each step (see
    # forward_layers). 0.2 or 0.5 are common; 0 (the default) is no dropout.
    def add_layer(self, num_inputs, num_neurons, act_fn, dropout=0.0):
        if act_fn not in ACTIVATIONS:
            raise ValueError(f"Invalid activation function: {act_fn}. Use one of {list(ACTIVATIONS)}")

        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be at least 0 and smaller than 1, got {dropout}")

        # The initial weights are random: if they were all equal, every neuron of the layer would compute the same
        # thing and receive the same update forever. Their scale aims to keep the variance of z similar from layer to
        # layer, so the signal neither explodes nor fades: He initialization (variance 2 / num_inputs) for ReLU layers
        # and LeCun initialization (variance 1 / num_inputs) for the others. The factor 2 of He makes up for a ReLU
        # zeroing about half of its outputs, so it keeps the variance exactly from one ReLU layer to the next; choosing
        # by the layer's own activation, as here, is the usual rule of thumb.
        if act_fn == "relu":
            std_dev = math.sqrt(2 / num_inputs)
        else:
            std_dev = math.sqrt(1 / num_inputs)

        layer = {
            "weights": np.random.normal(0, std_dev, (num_neurons, num_inputs)),  # one row per neuron
            "biases": np.zeros(num_neurons),
            "act_fn": act_fn,
        }

        if dropout > 0:
            layer["dropout"] = dropout

        self.layers.append(layer)

    # loss: the loss function to minimize (see losses.py), or None when the gradient comes from outside the model, as in
    #       PPO (see ppo.py), which then calls forward_layers, backward and update itself
    # optimizer: how the gradients update the parameters, "sgd", "momentum", "rmsprop" or "adam" (see optimizers.py)
    def configure(self, loss, optimizer="sgd"):
        if loss is not None:
            check_loss(loss)

            if self.layers:
                check_output_for_loss(self.layers[-1]["act_fn"], loss)  # e.g. a cross-entropy needs probabilities

        self.configs["loss"] = loss
        self.configs["optimizer"] = optimizer
        self.optimizer = make_optimizer(optimizer)  # a new optimizer starts without the state of the previous one

    # Fits min-max scaling to the training inputs. It is stored in the configs, so it is saved with the model and
    # every input (in train, predict and evaluate) is scaled with the same values, also after loading the model.
    def fit_normalization(self, inputs):
        min_vals, max_vals = minmax_stats(inputs)
        self.configs["normalization"] = {"method": "minmax", "min": min_vals, "max": max_vals}

    # Applies the fitted normalization, if any (models without it receive the inputs unchanged).
    def normalize(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        normalization = self.configs.get("normalization")

        if normalization is None:
            return inputs

        return apply_minmax(inputs, normalization["min"], normalization["max"])

    def predict(self, inputs):
        return self.forward(inputs)

    # Forward pass: the inputs go through every layer, from the first to the last.
    def forward(self, inputs):
        return self.forward_layers(self.normalize(inputs))

    # Forward pass over inputs that are already normalized. Keeps the inputs, z and output of every layer in
    # self.cache, because backward() needs them to compute the gradients.
    #
    # training: applies dropout (train_step sets it). Each input of a layer with dropout is set to 0 with probability
    # dropout, and the others are multiplied by 1 / (1 - dropout), so every input keeps its expected value and the
    # predictions, which use all the inputs as they are, see the same scale (inverted dropout). A different random half
    # (with dropout 0.5) of the inputs is missing in every step, so no neuron can rely on any one input being there:
    # the network has to spread what it learns, which reduces overfitting.
    def forward_layers(self, values, training=False):
        self.cache = []

        for layer in self.layers:
            inputs = values
            mask = None  # 0 for the dropped inputs, 1 / (1 - dropout) for the others

            if training and layer.get("dropout", 0) > 0:
                mask = (np.random.random(np.shape(inputs)) >= layer["dropout"]) / (1 - layer["dropout"])
                inputs = inputs * mask

            z, values = layer_forward(inputs, layer["weights"], layer["biases"], layer["act_fn"])
            self.cache.append({"inputs": inputs, "z": z, "output": values, "mask": mask})

        self.notify("on_forward")
        return values

    # Trains with mini-batch gradient descent.
    #
    # Each epoch goes through the whole dataset in batches of batch_size samples, with one train_step per batch. With
    # shuffle, every epoch uses a new random order: data that is ordered (e.g. grouped by class) would otherwise give
    # batches that each push the model towards a single class.
    #
    # validation_data: optional (inputs, targets) not used for training. Its loss (and accuracy, for classification)
    # is calculated after each epoch, to see whether the model generalizes or starts to overfit.
    #
    # patience: early stopping, with validation_data. The training stops when the validation loss hasn't improved for
    # patience epochs in a row, and the model goes back to the weights of the epoch with the lowest validation loss
    # (also when it runs all the epochs), and the optimizer to its state at that epoch. Past that epoch, the model memorizes the training data instead of learning
    # what generalizes (overfitting), so the number of epochs no longer has to be guessed.
    #
    # verbose: show a progress bar during each epoch, and a line with the loss every log_every epochs (and the last).
    #
    # Returns self.history: the loss of each epoch, the validation metrics and, with patience, the best epoch.
    def train(self, inputs, targets, epochs, learning_rate, batch_size=32, l1_lambda=0.0, l2_lambda=0.0,
              shuffle=True, verbose=False, validation_data=None, log_every=1, patience=None):
        if patience is not None and (validation_data is None or patience < 1):
            raise ValueError("patience needs validation_data and must be at least 1")

        check_output_for_loss(self.layers[-1]["act_fn"], self.configs["loss"])
        inputs, targets = self.prepare_dataset(inputs, targets)
        inputs = self.normalize(inputs)  # once for the whole dataset (train_step expects normalized inputs)
        num_batches = math.ceil(len(inputs) / batch_size)
        self.history = {"loss": [], "val_loss": [], "val_accuracy": []}
        start_time = time.time()
        progress_line = ""
        best_val_loss, best_epoch, best_layers, best_optimizer = math.inf, 0, None, None  # for early stopping
        stop = False

        for epoch in range(epochs):
            order = np.random.permutation(len(inputs)) if shuffle else np.arange(len(inputs))
            epoch_loss = 0

            for batch_count, batch_indices in enumerate(create_batches(order, batch_size), start=1):
                batch_inputs, batch_targets = inputs[batch_indices], targets[batch_indices]
                batch_loss = self.train_step(batch_inputs, batch_targets, learning_rate, l1_lambda, l2_lambda)
                epoch_loss += batch_loss * len(batch_inputs)  # weighted, because the last batch may be smaller

                if verbose:
                    filled = round(30 * batch_count / num_batches)
                    progress_bar = '#' * filled + '.' * (30 - filled)
                    progress_line = f"Epoch {epoch+1}/{epochs} [{progress_bar}] batch {batch_count}/{num_batches} - loss {format_number(batch_loss)}"
                    print("\r" + progress_line, end="", flush=True)  # the next one is written over it

            self.history["loss"].append(epoch_loss / len(inputs))
            summary = f"Epoch {epoch+1}/{epochs} - loss {format_number(self.history['loss'][-1])}"

            if validation_data is not None:
                val_loss, val_accuracy = self.evaluate(*validation_data)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)

                summary += f" - val_loss {format_number(val_loss)}"
                if val_accuracy is not None:
                    summary += f" - val_accuracy {val_accuracy:.4f}"

                # Early stopping: keep the weights of the best epoch, and stop after patience epochs without a better one
                if patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss, best_epoch, best_layers = val_loss, epoch + 1, self.get_layers()
                        best_optimizer = copy.deepcopy(self.optimizer)  # its state goes back with the weights
                    elif epoch + 1 - best_epoch >= patience:
                        stop = True

            self.notify("on_metrics", {"epoch": epoch + 1, "loss": self.history["loss"][-1],
                                       "val_loss": self.history["val_loss"][-1] if validation_data is not None else None,
                                       "val_accuracy": self.history["val_accuracy"][-1] if validation_data is not None else None})

            if verbose and ((epoch + 1) % log_every == 0 or epoch + 1 == epochs or stop):
                summary += f" - {timing(start_time, epoch + 1, epoch + 1 if stop else epochs)}"
                print("\r" + summary.ljust(len(progress_line)))  # replaces the progress bar of the epoch

            if stop:
                break

        if verbose:
            print("\r" + " " * len(progress_line), end="\r")  # clears a progress bar left by an epoch without summary

        if patience is not None and best_layers is not None:
            self.layers, self.optimizer = best_layers, best_optimizer  # copies of their own, kept at the best epoch
            self.history["best_epoch"] = best_epoch

            if verbose:
                best = f"{best_epoch} (val_loss {format_number(best_val_loss)})"
                if stop:
                    print(f"Early stopping at epoch {epoch + 1}: no better validation loss for {patience} epochs. "
                          f"The model goes back to the weights of epoch {best}.")
                else:
                    print(f"The model keeps the weights of its best epoch: {best}.")

        elif patience is not None and verbose:  # no epoch had a valid validation loss
            stopped = f"Early stopping at epoch {epoch + 1}: " if stop else ""
            print(f"{stopped}the validation loss was never a number (NaN or infinite): the training diverged, and the "
                  "model keeps the weights of its last epoch. A smaller learning rate usually prevents it.")

        return self.history

    # One step of gradient descent on a batch of (normalized) inputs:
    #   1. forward pass: the predictions for the batch
    #   2. loss: how wrong they are (the mean over the batch)
    #   3. output delta: the gradient of the loss with respect to z of the output layer
    #   4. backpropagation: the gradient of every weight and bias, from the last layer to the first
    #   5. update: every parameter takes a step of learning_rate against its gradient
    # Returns the loss of the batch (before the update).
    def train_step(self, inputs, targets, learning_rate, l1_lambda=0.0, l2_lambda=0.0):
        predictions = self.forward_layers(inputs, training=True)
        loss = loss_fn(self.configs["loss"], targets, predictions)
        output_delta = self.output_delta(targets, predictions)
        gradients = self.backward(output_delta, l1_lambda, l2_lambda)
        self.update(gradients, learning_rate)
        return loss

    # Gradient of the loss with respect to z of the output layer, called delta: one row per sample of the batch.
    def output_delta(self, targets, predictions):
        output_layer = self.layers[-1]

        # Softmax with cross-entropy: the chain rule simplifies to (predictions - targets) / batch_size
        if is_softmax_crossentropy(output_layer["act_fn"], self.configs["loss"]):
            return softmax_crossentropy_delta(self.configs["loss"], targets, predictions)

        # Sigmoid with binary cross-entropy: likewise, (predictions - targets) / (batch_size * num_outputs)
        if is_sigmoid_binary_crossentropy(output_layer["act_fn"], self.configs["loss"]):
            return sigmoid_binary_crossentropy_delta(targets, predictions)

        # Otherwise, the derivative of the loss gives dL/dy (clipped, see clip_gradients), and the chain rule through
        # the activation gives dL/dz = dL/dy * f'(z)
        grad_output = clip_gradients(loss_derivative_fn(self.configs["loss"], targets, predictions), 10.0)
        cache = self.cache[-1]
        return activation_backward(output_layer["act_fn"], cache["z"], cache["output"], grad_output)

    # Backpropagation: the chain rule applied from the last layer to the first. For a layer whose neurons have the
    # gradient delta = dL/dz (one row per sample):
    #   dL/dweights[j, k] = sum over the samples of delta[j] * inputs[k]   (each unit of weight changes z[j] by inputs[k])
    #   dL/dbiases[j]     = sum over the samples of delta[j]
    #   dL/dinputs[k]     = sum over the neurons of delta[j] * weights[j, k]
    # The inputs of a layer are the outputs of the previous one, so dL/dinputs goes through the activation of the
    # previous layer (activation_backward) to become its delta, and so on down to the first layer. Summing over the
    # samples gives the gradient of the mean loss, because delta already includes the division by the batch size.
    #
    # Uses the values kept in self.cache by the last forward pass. Returns (dL/dweights, dL/dbiases) for each layer.
    def backward(self, output_delta, l1_lambda=0.0, l2_lambda=0.0):
        gradients = [None] * len(self.layers)
        delta = np.atleast_2d(output_delta)  # shape (batch_size, num_neurons)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_inputs = np.atleast_2d(self.cache[i]["inputs"])

            grad_weights = np.dot(delta.T, layer_inputs)  # shape (num_neurons, num_inputs), like the weights
            grad_biases = np.sum(delta, axis=0)

            # Regularization adds the gradient of a penalty on large weights, l1 * sum(|w|) + l2 / 2 * sum(w^2), which
            # pulls the weights towards 0 and reduces overfitting (the biases aren't penalized)
            if l1_lambda > 0:
                grad_weights += l1_lambda * np.sign(layer["weights"])

            if l2_lambda > 0:
                grad_weights += l2_lambda * layer["weights"]

            gradients[i] = (grad_weights, grad_biases)

            if i > 0:
                previous = self.cache[i - 1]
                grad_inputs = np.dot(delta, layer["weights"])

                # A dropped input didn't reach the layer, so it gets no gradient; the others get the same scale
                if self.cache[i].get("mask") is not None:
                    grad_inputs = grad_inputs * self.cache[i]["mask"]

                delta = np.atleast_2d(activation_backward(self.layers[i - 1]["act_fn"], previous["z"], previous["output"], grad_inputs))

        return gradients

    # Gradient descent: the gradient points to where the loss grows the fastest, so every parameter takes a small step
    # the other way. The learning rate sets the size of the step: too small learns slowly, too large overshoots.
    # The optimizer decides the exact step: proportional to the gradient (SGD), or adapted to each parameter (Adam).
    def update(self, gradients, learning_rate):
        self.optimizer.update(self.layers, gradients, learning_rate)
        self.notify("on_update", gradients, learning_rate)

    def summary(self):
        print()
        print("Model Summary")
        print("--------------")

        total_params = 0

        for i, layer in enumerate(self.layers):
            num_neurons, num_inputs = layer["weights"].shape
            layer_params = num_neurons * num_inputs + num_neurons  # one weight per input and one bias, per neuron
            total_params += layer_params

            print(f"Layer {i+1}:")
            print(f"    Activation Function: {layer['act_fn']}")
            print(f"    Number of Neurons: {num_neurons}")
            print(f"    Input Dimension: {num_inputs}")
            print(f"    Number of Parameters: {layer_params}")
            if layer.get("dropout"):
                print(f"    Dropout of the inputs: {layer['dropout']}")
            print("--------------")

        print(f"Total Parameters: {total_params}")
        print(f"Loss Function: {self.configs['loss']}")
        print(f"Optimizer: {self.configs.get('optimizer', 'sgd')}")

        if "normalization" in self.configs:
            print(f"Input Normalization: {self.configs['normalization']['method']}")
        print()

    # Brings a dataset to the shapes the training expects: inputs (num_samples, num_inputs) and targets
    # (num_samples, num_outputs), or class indices for sparse_categorical_crossentropy.
    #
    # A flat list of values is accepted for a model with one input (inputs) or one output (targets). Any other
    # mismatch raises an error, because NumPy broadcasting would otherwise produce a wrong loss without failing.
    def prepare_dataset(self, inputs, targets):
        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets)
        num_inputs = self.layers[0]["weights"].shape[1]
        num_outputs = self.layers[-1]["weights"].shape[0]

        if inputs.ndim == 1 and num_inputs == 1:
            inputs = inputs.reshape(-1, 1)

        if inputs.ndim != 2 or inputs.shape[1] != num_inputs:
            raise ValueError(f"inputs must have shape (num_samples, {num_inputs}), got {inputs.shape}")

        if self.configs["loss"] == "sparse_categorical_crossentropy":
            expected_shapes = [(len(inputs),), (len(inputs), 1)]
        else:
            if targets.ndim == 1 and num_outputs == 1:
                targets = targets.reshape(-1, 1)

            expected_shapes = [(len(inputs), num_outputs)]

        if targets.shape not in expected_shapes:
            raise ValueError(f"targets must have shape {' or '.join(map(str, expected_shapes))}, got {targets.shape}")

        return inputs, targets

    # Returns the mean loss and the accuracy (None for models that aren't classifiers, see accuracy).
    def evaluate(self, test_inputs, test_labels):
        test_inputs, test_labels = self.prepare_dataset(test_inputs, test_labels)
        predictions = self.forward(test_inputs)
        loss = loss_fn(self.configs["loss"], test_labels, predictions)

        return loss, self.accuracy(predictions, test_labels)

    # Fraction of correct predictions, for classification models:
    # - binary_crossentropy: each output is a yes/no answer, yes above 0.5; the fraction of the answers that are right
    # - class indices (sparse_categorical_crossentropy), softmax or several sigmoid outputs: the output with the
    #   highest value is the predicted class
    # - a single sigmoid output (binary classification): above 0.5 is class 1
    # Other outputs (e.g. linear, for regression) have no accuracy, so it returns None (see regression_metrics).
    def accuracy(self, predictions, labels):
        act_fn = self.layers[-1]["act_fn"]

        if self.configs["loss"] == "binary_crossentropy":
            predicted = predictions > 0.5
            expected = labels.reshape(predictions.shape) > 0.5
        elif self.configs["loss"] == "sparse_categorical_crossentropy":
            predicted = np.argmax(predictions, axis=-1)
            expected = labels.astype(int).reshape(predicted.shape)  # labels are class indices
        elif act_fn == "softmax" or (act_fn == "sigmoid" and predictions.shape[-1] > 1):
            predicted = np.argmax(predictions, axis=-1)
            expected = np.argmax(labels, axis=-1)  # labels are one-hot encoded
        elif act_fn == "sigmoid":
            predicted = predictions > 0.5
            expected = labels > 0.5
        else:
            return None

        return float(np.mean(predicted == expected))

    # Error metrics for regression models, in the units of the targets:
    # - mae: mean absolute error
    # - rmse: root mean squared error (penalizes large errors more than mae)
    # - r2: coefficient of determination (1 is perfect, 0 is as good as always predicting the mean of the targets);
    #   with several outputs, their errors and variances are added up before dividing
    def regression_metrics(self, test_inputs, test_targets):
        test_inputs, test_targets = self.prepare_dataset(test_inputs, test_targets)
        errors = self.forward(test_inputs) - test_targets
        total_variance = np.sum((test_targets - np.mean(test_targets, axis=0)) ** 2)

        return {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "r2": float(1 - np.sum(errors ** 2) / total_variance) if total_variance > 0 else float("nan"),
        }

    # Saves the parameters of each layer and the configs (not the cache of the last forward pass).
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        dump({"layers": self.layers, "configs": self.configs}, filepath)

    def get_layers(self):
        return copy.deepcopy(self.layers)

    def set_layers(self, layers):
        self.layers = copy.deepcopy(layers)

    def get_configs(self):
        return copy.deepcopy(self.configs)
