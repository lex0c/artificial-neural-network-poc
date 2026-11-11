import math
import random
import copy
import numpy as np
from joblib import dump, load

from etc import activation_fn, activation_backward, loss_fn, loss_derivative_fn, clip_gradients, normalize_gradients, create_batches, softmax_crossentropy_delta, is_softmax_crossentropy, minmax_stats, apply_minmax


def layer(values, weights, biases, act_fn):
    # Linear combination of every neuron at once (each row of weights is one neuron):
    # (batch_size, num_inputs) x (num_inputs, num_neurons) -> (batch_size, num_neurons)
    # A single sample with shape (num_inputs,) gives shape (num_neurons,).
    z = np.dot(values, weights.T) + biases  # biases will be broadcast over the batch

    # The activation function sees the whole layer, so softmax can normalize across the neurons of each sample
    return z, activation_fn(act_fn, z)


def load_model(filepath):
    model_params = load(filepath)
    return FeedForward(layers=model_params["layers"], configs=model_params["configs"])


def clone_model(model):
    return FeedForward(layers=model.get_layers(), configs=model.get_configs())


class FeedForward:
    def __init__(self, layers=None, configs=None):
        # Default to new containers: a mutable default argument would be shared by every model
        self.layers = layers if layers is not None else []
        self.configs = configs if configs is not None else {}
        self.learning_rate = 0.0001
        self.history = {}

    def add_layer(self, num_inputs, num_neurons, act_fn):
        if act_fn == "relu":
            std_dev = math.sqrt(2 / num_inputs) # He initialization for ReLU
        else:
            std_dev = math.sqrt(1 / num_inputs) # Xavier initialization for other activations

        weights = np.random.normal(0, std_dev, (num_neurons, num_inputs))
        biases = np.zeros(num_neurons)

        self.layers.append({
            "weights": weights,
            "biases": biases,
            "act_fn": act_fn,
            "z": [],
            "output": []
        })

    def configure(self, loss):
        self.configs["loss"] = loss

    # Fits min-max scaling to the training inputs. It is stored in the configs, so it is saved with the model
    # and every input (in train, predict and evaluate) is scaled with the same values, including after loading.
    def fit_normalization(self, inputs):
        min_vals, max_vals = minmax_stats(inputs)
        self.configs["normalization"] = {"method": "minmax", "min": min_vals, "max": max_vals}

    # Applies the fitted normalization, if any (models without it receive the inputs unchanged).
    def normalize(self, input_values):
        values = np.asarray(input_values, dtype=float)
        normalization = self.configs.get("normalization")

        if normalization is None:
            return values

        return apply_minmax(values, normalization["min"], normalization["max"])

    def forward(self, input_values, verbose=False):
        return self.forward_layers(self.normalize(input_values), verbose)

    # Forward pass over inputs that are already normalized
    def forward_layers(self, values, verbose=False):
        layer_size = len(self.layers)

        for i, layer_param in enumerate(self.layers):
            z, values = layer(values, layer_param["weights"], layer_param["biases"], layer_param["act_fn"])
            layer_param["z"] = z  # pre-activation, where the activation derivatives are evaluated
            layer_param["output"] = values

            if verbose:
                progress_bar = '#' * (i+1) + '.' * (layer_size - (i+1))
                print(f"> {i+1}/{layer_size} [{progress_bar}] layer step", end='\r', flush=True)

        return values

    # validation_data: optional (inputs, targets) not used for training. Its loss (and accuracy, for classification)
    # is calculated after each epoch, to see whether the model generalizes or starts to overfit.
    #
    # The loss of each epoch (and the validation metrics) are kept in self.history.
    def train(self, inputs, targets, epochs, learning_rate, batch_size=32, l1_lambda=0.0, l2_lambda=0.0, verbose=False, validation_data=None):
        self.learning_rate = learning_rate
        self.history = {"loss": [], "val_loss": [], "val_accuracy": []}
        total_loss = 0

        inputs, targets = self.prepare_dataset(inputs, targets)

        # Normalized once for the whole dataset; the backward pass needs the same normalized inputs as the forward pass
        inputs = self.normalize(inputs)
        num_batches = math.ceil(len(inputs) / batch_size)  # the last batch may be smaller

        for epoch in range(epochs):
            epoch_loss = 0

            for batch_count, (batch_inputs, batch_targets) in enumerate(create_batches(inputs, targets, batch_size), start=1):
                predictions = self.forward_layers(batch_inputs)

                # Calculates the mean loss of the batch
                batch_loss = loss_fn(self.configs["loss"], batch_targets, predictions)

                # Weighted by the number of samples, so a smaller last batch doesn't count as a full one
                epoch_loss += batch_loss * len(batch_inputs)

                if is_softmax_crossentropy(self.layers[-1]["act_fn"], self.configs["loss"]):
                    # Gradient in relation to the pre-activation of the softmax layer, already bounded to [-1, 1]
                    gradients = softmax_crossentropy_delta(self.configs["loss"], batch_targets, predictions)
                    skip_last_activation = True
                else:
                    # Calculates the gradient of the mean batch loss in relation to the output
                    gradients = loss_derivative_fn(self.configs["loss"], batch_targets, predictions)
                    skip_last_activation = False

                    # Clip and normalize gradients
                    gradients = clip_gradients(gradients, 10.0)
                    #gradients = normalize_gradients(gradients)

                # Backpropagation: one update per batch, with the gradient averaged over its samples
                self.backward(gradients, batch_inputs, l1_lambda, l2_lambda, skip_last_activation)

                if verbose:
                    filled = round(30 * batch_count / num_batches)
                    progress_bar = '#' * filled + '.' * (30 - filled)
                    print(f"> Epoch: {epoch+1}/{epochs}, Batch: {batch_count}/{num_batches} [{progress_bar}] - batch_loss: {batch_loss}", end='\r', flush=True)

            epoch_loss /= len(inputs)
            total_loss += epoch_loss
            self.history["loss"].append(epoch_loss)

            summary = f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_loss}"

            if validation_data is not None:
                val_loss, val_accuracy = self.evaluate(*validation_data)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)

                summary += f", Val Loss: {val_loss}"
                if val_accuracy is not None:
                    summary += f", Val Accuracy: {val_accuracy:.4f}"

            print('')  # add an empty line to prevent flush
            print(f"{summary}, LR: {self.learning_rate}")

        return total_loss / epochs, self.learning_rate

    # Backpropagation: goes from the last layer to the first, calculating the gradients of each layer with matrix
    # operations and updating its weights and biases.
    #
    # gradients: gradient of the mean batch loss in relation to the output of the last layer
    #            (or to its pre-activation, when skip_last_activation is True).
    # input_values: the normalized batch that was passed to forward_layers().
    def backward(self, gradients, input_values, l1_lambda=0.0, l2_lambda=0.0, skip_last_activation=False):
        gradients = np.atleast_2d(gradients)  # shape (batch_size, num_neurons)
        last = len(self.layers) - 1

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # If it's the first layer, the input comes directly from the dataset
            previous_output = np.atleast_2d(self.layers[i-1]["output"] if i > 0 else input_values)

            # Gradient of the loss in relation to each neuron's pre-activation, per sample: delta[s, j]
            if i == last and skip_last_activation:
                delta = gradients
            else:
                delta = np.atleast_2d(activation_backward(layer["act_fn"], layer["z"], layer["output"], gradients))

            # Weight and bias gradients, summed over the samples of the batch
            # (the loss gradient already carries the 1/batch_size factor):
            #   grad_weight[j, k] = sum over samples s of delta[s, j] * previous_output[s, k]
            #   grad_bias[j]      = sum over samples s of delta[s, j]
            grad_weight = np.dot(delta.T, previous_output)  # shape (num_neurons, num_inputs)
            grad_bias = np.sum(delta, axis=0)               # one value per neuron

            # L1 regularization
            if l1_lambda > 0:
                grad_weight += l1_lambda * np.sign(layer["weights"])

            # L2 regularization
            if l2_lambda > 0:
                grad_weight += l2_lambda * layer["weights"]

            # Prepares the gradient for the next layer (previous in the order of execution), using the weights from
            # the forward pass, before they are updated:
            #   gradients[s, k] = sum over neurons j of delta[s, j] * weights[j, k]
            if i > 0:
                gradients = np.dot(delta, layer["weights"])

            # Updates weights and biases
            layer["weights"] -= self.learning_rate * grad_weight
            layer["biases"] -= self.learning_rate * grad_bias

    def predict(self, input_data, verbose=False):
        return self.forward(input_data, verbose)

    def summary(self):
        print("\n")
        print("Model Summary")
        print("--------------")

        total_params = 0

        for i, layer in enumerate(self.layers):
            # Calculate the number of parameters in the current layer
            num_inputs = len(layer["weights"][0])  # Number of inputs to the neurons in this layer
            num_neurons = len(layer["weights"])    # Number of neurons in this layer
            num_weights = num_inputs * num_neurons # Total weights in the layer
            num_biases = num_neurons               # One bias per neuron
            layer_params = num_weights + num_biases

            # Update total parameters count
            total_params += layer_params

            # Print layer information
            print(f"Layer {i+1}:")
            print(f"    Activation Function: {layer['act_fn']}")
            print(f"    Number of Neurons: {num_neurons}")
            print(f"    Input Dimension: {num_inputs}")
            print(f"    Number of Parameters: {layer_params}")
            print("--------------")

        print(f"Total Parameters: {total_params}")
        print(f"Loss Function: {self.configs['loss']}")

        if "normalization" in self.configs:
            print(f"Input Normalization: {self.configs['normalization']['method']}")
        print("\n")

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

    def evaluate(self, test_inputs, test_labels):
        test_inputs, test_labels = self.prepare_dataset(test_inputs, test_labels)
        predictions = self.forward(test_inputs)

        # Calculate mean loss
        loss = loss_fn(self.configs["loss"], test_labels, predictions)

        return loss, self.accuracy(predictions, test_labels)

    # Fraction of correct predictions, for classification models:
    # - class indices (sparse_categorical_crossentropy), softmax or several sigmoid outputs: the output with the
    #   highest value is the predicted class
    # - a single sigmoid output (binary classification): above 0.5 is class 1
    # Other outputs (e.g. linear, for regression) have no accuracy, so it returns None (see regression_metrics).
    def accuracy(self, predictions, labels):
        act_fn = self.layers[-1]["act_fn"]

        if self.configs["loss"] == "sparse_categorical_crossentropy":
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
    # - r2: coefficient of determination (1 is perfect, 0 is as good as always predicting the mean of the targets)
    def regression_metrics(self, test_inputs, test_targets):
        test_inputs, test_targets = self.prepare_dataset(test_inputs, test_targets)
        errors = self.forward(test_inputs) - test_targets
        total_variance = np.sum((test_targets - np.mean(test_targets, axis=0)) ** 2)

        return {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "r2": float(1 - np.sum(errors ** 2) / total_variance) if total_variance > 0 else float("nan"),
        }

    def save(self, filepath):
        # Only the parameters: the activations of the last forward pass ("z" and "output") aren't needed to use the
        # model, and their size depends on whatever batch ran last
        layers = [{"weights": l["weights"], "biases": l["biases"], "act_fn": l["act_fn"]} for l in self.layers]
        dump({"layers": layers, "configs": self.configs}, filepath)

    def get_layers(self):
        return copy.deepcopy(self.layers)

    def set_layers(self, layers):
        self.layers = copy.deepcopy(layers)

    def get_configs(self):
        return copy.deepcopy(self.configs)

