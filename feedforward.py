import math
import random

from etc import relu, sigmoid, softmax, linear, mse_loss_derivative, mse_loss, activation_derivative, clip_gradients


def neuron(values, weights, bias, act_fn):
    if len(values) != len(weights):
        raise ValueError("Sizes of values and weights must be the same!")

    linear_combination = 0
    output = 0

    # Linear combination of values and weights
    for value, weight in zip(values, weights):
        linear_combination += value * weight

    # Add bias
    linear_combination += bias

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
    if len(weights) != len(biases):
        raise ValueError("Sizes of weights and biases must be the same!")

    # Number of neurons
    size = len(weights)

    # Init empty array
    output = [0]*size

    # Run neurons
    for i in range(size):
        output[i] = neuron(values, weights[i], biases[i], act_fn)

        if verbose:
            print(f'Neuron {i+1}: {output[i]}')

    return output


class FeedForward:
    def __init__(self, layers=[], verbose=False):
        self.layers = layers
        self.verbose = verbose
        self.learning_rate = 0.01 # Default learning rate, can be set explicitly

    def add_layer(self, num_inputs, num_neurons, act_fn):
        limit = 1 / math.sqrt(num_inputs)

        self.layers.append({
            # The code below generates a list of lists, where each internal list represents the weights for a neuron in the neural network layer.
            "weights": [[random.uniform(-limit, limit) for _ in range(num_inputs)] for _ in range(num_neurons)],
            "biases": [random.uniform(-limit, limit) for _ in range(num_neurons)],
            "act_fn": act_fn,
            "output": [],
            "delta": [0] * num_neurons # Initialize delta values as zeros
        })

    def forward(self, input_values):
        values = input_values

        for i, layer_param in enumerate(self.layers):
            if self.verbose:
                print(f'Processing Layer {i+1}')

            values = layer(values, layer_param["weights"], layer_param["biases"], layer_param["act_fn"], self.verbose)
            layer_param['output'] = values

        return values

    def backward(self, target):
        # Calculate error at output
        error = mse_loss_derivative(self.layers[-1]['output'], target)

        # Propagate the error backward
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            outputs = layer['output']
            act_fn = layer['act_fn']
            deltas = []

            for j, output in enumerate(outputs):
                # Calculate delta
                delta = error[j] * activation_derivative(act_fn, output)
                delta = clip_gradients(delta, max_value=1.0)
                deltas.append(delta)

                # Update weights and biases
                for k in range(len(layer['weights'][j])):
                    # Gradient for weight is delta * input to the neuron
                    grad_w = delta * (self.layers[i-1]['output'][k] if i > 0 else 1)
                    layer['weights'][j][k] -= self.learning_rate * grad_w

                # Gradient for bias is simply the delta
                layer['biases'][j] -= self.learning_rate * delta

            layer['delta'] = deltas # Store deltas in layer parameters for possible future use

            # Prepare error for next layer (if not the first layer)
            if i > 0:
                new_error = [0] * len(self.layers[i-1]['output'])

                for j in range(len(layer['weights'])):
                    for k in range(len(layer['weights'][j])):
                        new_error[k] += deltas[j] * layer['weights'][j][k]

                error = new_error

    def train(self, data, targets, epochs, learning_rate):
        self.learning_rate = learning_rate # Set the learning rate for this training session

        for epoch in range(epochs):
            for input_values, target in zip(data, targets):
                outputs = self.forward(input_values)

                self.backward(target)

                loss = mse_loss(outputs, target)

                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

        print('Training completed')


