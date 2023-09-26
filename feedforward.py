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
            "biases": [0.0 for _ in range(num_neurons)],
            "act_fn": act_fn
        })

    def forward(self, input_values):
        values = input_values

        for i, layer_param in enumerate(self.layers):
            if self.verbose:
                print(f'Processing Layer {i+1}')

            values = layer(values, layer_param["weights"], layer_param["biases"], layer_param["act_fn"], self.verbose)

        return values

    def select_action(self, state, epsilon=0.1, threshold=0.5):
        if random.uniform(0, 1) < epsilon:
            return random.choice([0, 1])

        layer_output = self.forward(state)

        if layer_output:
            return 1 if layer_output[0] >= threshold else 0

        return random.choice([0, 1])

    def learn(self, state, action, reward, next_state, done, learning_rate=0.1, discount_factor=0.99):
        # Predict Q-values for current state
        q_values = self.forward(state)

        # Copy Q-values as Target Q-values
        target_q_values = list(q_values)

        if done:
            target_q_values[0] = reward # Assume the reward is correctly calculated externally
        else:
            # Predict Q-values for the next state
            next_q_values = self.forward(next_state)

            # Update the target Q-value for the action taken
            target_q_values[0] = reward + discount_factor * max(next_q_values)

        # Compute the error
        error = [t - q for t, q in zip(target_q_values, q_values)]

        # Update weights and biases
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            for neuron_idx, weights in enumerate(layer["weights"]):
                for weight_idx, weight in enumerate(weights):
                    weights[weight_idx] += learning_rate * error[0] * state[weight_idx]

                layer["biases"][neuron_idx] += learning_rate * error[0]


