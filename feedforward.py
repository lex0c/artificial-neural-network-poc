import math
import random
import numpy as np
from joblib import dump, load

from etc import relu, sigmoid, softmax, linear, mse_loss_derivative, mse_loss, activation_derivative


def neuron(values, weights, bias, act_fn):
    # Matrix multiplication to handle batch processing
    linear_combination = np.dot(values, weights) + bias  # bias will be broadcast

    output = 0

    # Pass the linear combination through the activation function
    if act_fn == "relu":
        output = relu(linear_combination)
    elif act_fn == "sigmoid":
        output = sigmoid(linear_combination)
    elif act_fn == "softmax":
        output = softmax(linear_combination)
    elif act_fn == "linear":
        output = linear(linear_combination)

    return output


def layer(values, weights, biases, act_fn, verbose=False):
    outputs = []

    for i in range(len(weights)):  # Iterate over each neuron
        output = neuron(values, weights[i], biases[i], act_fn)
        outputs.append(output)

    # Stack outputs to maintain batch structure
    outputs = np.stack(outputs, axis=-1)  # shape (batch_size, num_neurons)

    return outputs


def load_model(filepath):
    return FeedForward(layers=load(filepath))


class FeedForward:
    def __init__(self, layers=[], verbose=False):
        self.layers = layers
        self.verbose = verbose
        self.learning_rate = 0.001

    def add_layer(self, num_inputs, num_neurons, act_fn):
        if act_fn == "relu":
            limit = math.sqrt(2 / num_inputs) # He initialization for ReLU
        else:
            limit = 1 / math.sqrt(num_inputs) # Xavier initialization for other activations

        self.layers.append({
            # The code below generates a list of lists, where each internal list represents the weights for a neuron in the neural network layer.
            "weights": [[random.uniform(-limit, limit) for _ in range(num_inputs)] for _ in range(num_neurons)],
            "biases": [random.uniform(-limit, limit) for _ in range(num_neurons)],
            "act_fn": act_fn,
            "output": []
        })

    def forward(self, input_values):
        values = input_values

        for i, layer_param in enumerate(self.layers):
            if self.verbose:
                print(f"Processing Layer {i+1}")

            values = layer(values, layer_param["weights"], layer_param["biases"], layer_param["act_fn"], self.verbose)
            layer_param["output"] = values

        return values

    def train(self, inputs, targets, epochs, learning_rate):
        self.learning_rate = learning_rate

        for epoch in range(epochs):
            total_loss = 0

            for input_values, target in zip(inputs, targets):
                predictions = self.forward(input_values)

                # Calculates the loss
                loss = mse_loss(target, predictions)
                total_loss += loss

                # Calculates the gradient of the loss in relation to the output
                gradients = mse_loss_derivative(target, predictions)

                # Backpropagation
                self.backward(gradients, input_values)

            print(f"Epoch: {epoch+1}, Loss: {total_loss / len(inputs)}")

    def backward(self, gradients, input_values):
        # Propagates the output gradient backwards
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            incoming_gradients = []

            # Calculates the gradient for each neuron in the layer
            for j in range(len(layer["weights"])):
                # Derivative of the activation function
                d_activation = activation_derivative(layer["act_fn"], layer["output"][j])

                # Gradient of the error in relation to the neuron's output
                delta = gradients[j] * d_activation

                # Update weights and bias
                for k in range(len(layer["weights"][j])):
                    # If it's the first layer, the input comes directly from the dataset
                    if i == 0:
                        input_to_use = input_values[k]
                    else:
                        input_to_use = self.layers[i-1]["output"][k]

                    # Weight gradient
                    grad_weight = delta * input_to_use
                    layer["weights"][j][k] -= self.learning_rate * grad_weight

                # Bias gradient
                grad_bias = delta
                layer["biases"][j] -= self.learning_rate * grad_bias

                # Accumulates the gradient for the next layer
                if i > 0:
                    for k in range(len(layer["weights"][j])):
                        if len(incoming_gradients) <= k:
                            incoming_gradients.append(0)

                        incoming_gradients[k] += layer["weights"][j][k] * delta

            gradients = incoming_gradients

    def predict(self, input_data):
        return self.forward(input_data)

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

        # Print the total parameters at the end
        print(f"Total Parameters: {total_params}")
        print("\n")

    def evaluate(self, test_inputs, test_labels):
        return None, None

    def save(self, filepath):
            dump(self.layers, filepath)

