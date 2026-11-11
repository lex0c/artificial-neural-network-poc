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

    def train(self, inputs, targets, epochs, learning_rate, batch_size=32, l1_lambda=0.0, l2_lambda=0.0, use_granular_update=False, verbose=False):
        self.learning_rate = learning_rate
        total_loss = 0

        # Normalized once for the whole dataset; the backward pass needs the same normalized inputs as the forward pass
        inputs = self.normalize(inputs)
        targets = np.asarray(targets)
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
                if use_granular_update:
                    self.backward_stable(gradients, batch_inputs, skip_last_activation)
                else:
                    self.backward_unstable(gradients, batch_inputs, l1_lambda, l2_lambda, skip_last_activation)

                if verbose:
                    filled = round(30 * batch_count / num_batches)
                    progress_bar = '#' * filled + '.' * (30 - filled)
                    print(f"> Epoch: {epoch+1}/{epochs}, Batch: {batch_count}/{num_batches} [{progress_bar}] - batch_loss: {batch_loss}", end='\r', flush=True)

            epoch_loss /= len(inputs)
            total_loss += epoch_loss

            print('')  # add an empty line to prevent flush
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_loss}, LR: {self.learning_rate}")

        return total_loss / epochs, self.learning_rate

    # Updates each weight individually (slower, but easier to follow)
    #
    # gradients: gradient of the mean batch loss in relation to the output of the last layer
    #            (or to its pre-activation, when skip_last_activation is True).
    # input_values: the batch that was passed to forward().
    def backward_stable(self, gradients, input_values, skip_last_activation=False):
        gradients = np.atleast_2d(gradients)  # shape (batch_size, num_neurons)
        last = len(self.layers) - 1

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            weights = layer["weights"]

            # If it's the first layer, the input comes directly from the dataset
            previous_output = np.atleast_2d(self.layers[i-1]["output"] if i > 0 else input_values)

            # Gradient of the loss in relation to each neuron's pre-activation, per sample
            if i == last and skip_last_activation:
                delta = gradients
            else:
                delta = np.atleast_2d(activation_backward(layer["act_fn"], layer["z"], layer["output"], gradients))

            incoming_gradients = np.zeros_like(previous_output)

            # Calculates the gradient for each neuron in the layer
            for j in range(len(weights)):
                for k in range(len(weights[j])):
                    # Accumulates the gradient for the previous layer, using the weight from the forward pass
                    # (read before it is updated below)
                    if i > 0:
                        incoming_gradients[:, k] += weights[j][k] * delta[:, j]

                    # Weight gradient, summed over the samples of the batch
                    grad_weight = np.sum(delta[:, j] * previous_output[:, k])
                    weights[j][k] -= self.learning_rate * grad_weight

                # Bias gradient
                grad_bias = np.sum(delta[:, j])
                layer["biases"][j] -= self.learning_rate * grad_bias

            gradients = incoming_gradients

    # Updates all the weights of a layer at once with matrix operations (faster)
    #
    # gradients: gradient of the mean batch loss in relation to the output of the last layer
    #            (or to its pre-activation, when skip_last_activation is True).
    # input_values: the batch that was passed to forward().
    def backward_unstable(self, gradients, input_values, l1_lambda=0.0, l2_lambda=0.0, skip_last_activation=False):
        gradients = np.atleast_2d(gradients)  # shape (batch_size, num_neurons)
        last = len(self.layers) - 1

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # If it's the first layer, the input comes directly from the dataset
            previous_output = np.atleast_2d(self.layers[i-1]["output"] if i > 0 else input_values)

            # Gradient of the loss in relation to each neuron's pre-activation, per sample
            if i == last and skip_last_activation:
                delta = gradients
            else:
                delta = np.atleast_2d(activation_backward(layer["act_fn"], layer["z"], layer["output"], gradients))

            # Weight and bias gradients, summed over the samples of the batch
            # (the loss gradient already carries the 1/batch_size factor)
            grad_weight = np.dot(delta.T, previous_output)  # shape (num_neurons, num_inputs)
            grad_bias = np.sum(delta, axis=0)               # one value per neuron

            # L1 regularization
            if l1_lambda > 0:
                grad_weight += l1_lambda * np.sign(layer["weights"])

            # L2 regularization
            if l2_lambda > 0:
                grad_weight += l2_lambda * layer["weights"]

            # Prepares the gradient for the next layer (previous in the order of execution),
            # using the weights from the forward pass, before they are updated
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

    def evaluate(self, test_inputs, test_labels):
        test_labels = np.asarray(test_labels)
        predictions = self.forward(test_inputs)

        # Calculate mean loss
        loss = loss_fn(self.configs["loss"], test_labels, predictions)

        # Calculate accuracy
        predicted_classes = np.argmax(predictions, axis=-1)

        if self.configs["loss"] == "sparse_categorical_crossentropy":
            true_classes = test_labels.astype(int).reshape(predicted_classes.shape)  # labels are class indices
        else:
            true_classes = np.argmax(test_labels, axis=-1)  # labels are one-hot encoded

        accuracy = np.mean(predicted_classes == true_classes)

        return loss, accuracy

    def save(self, filepath):
        dump({"layers": self.layers, "configs": self.configs}, filepath)

    def get_layers(self):
        return copy.deepcopy(self.layers)

    def set_layers(self, layers):
        self.layers = copy.deepcopy(layers)

    def get_configs(self):
        return copy.deepcopy(self.configs)

