import math
import random
#import numpy as np

from etc import relu, sigmoid, softmax, linear


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

    def add_layer(self, num_inputs, num_neurons, act_fn):
        limit = 1 / math.sqrt(num_inputs)

        self.layers.append({
            # The code below generates a list of lists, where each internal list represents the weights for a neuron in the neural network layer.
            "weights": [[random.uniform(-limit, limit) for _ in range(num_inputs)] for _ in range(num_neurons)],
            "biases": [random.uniform(-limit, limit) for _ in range(num_neurons)],
            "act_fn": act_fn
        })

    def forward(self, input_values):
        values = input_values

        for i, layer_param in enumerate(self.layers):
            if self.verbose:
                print(f'Processing Layer {i+1}')

            values = layer(values, layer_param["weights"], layer_param["biases"], layer_param["act_fn"], self.verbose)

        return values

