import math
import random
import numpy as np

from etc import relu, sigmoid, softmax, linear, activation_derivative, mse_loss, mse_loss_derivative


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
            "act_fn": act_fn,
            "output": []
        })

    def forward(self, input_values):
        values = input_values

        for i, layer_param in enumerate(self.layers):
            if self.verbose:
                print(f'Processing Layer {i+1}')

            values = layer(values, layer_param["weights"], layer_param["biases"], layer_param["act_fn"], self.verbose)

            layer_param['output'] = values

        return values


class FeedForwardAutoencoder(FeedForward):
    def __init__(self, layers=[], learning_rate=0.01, verbose=False):
        super().__init__(layers, verbose)
        self.learning_rate = learning_rate

    def backpropagation(self, X, loss_gradient):
        gradient = loss_gradient

        for i in reversed(range(len(self.layers))):
            layer_param = self.layers[i]

            # A entrada para a camada atual é a saída da camada anterior
            if i != 0:
                X = self.layers[i-1]['output']

            # Calculando o gradiente da função de ativação
            act_derivative = activation_derivative(layer_param['act_fn'], layer_param['output'])
            gradient *= act_derivative

            # Calculando o gradiente dos pesos e bias
            weights_gradient = np.outer(gradient, X)
            biases_gradient = gradient

            # Atualizando os pesos e bias
            layer_param['weights'] -= self.learning_rate * weights_gradient.T
            layer_param['biases'] -= self.learning_rate * biases_gradient

            # Atualizando o gradiente para a próxima iteração
            gradient = np.dot(layer_param['weights'], gradient)

    def train(self, X):
        # Forward pass
        reconstructed_X = self.forward(X)

        # Calculate loss
        loss = mse_loss(X, reconstructed_X)

        # Calculate loss gradient
        loss_gradient = mse_loss_derivative(X, reconstructed_X)

        # Backward pass
        self.backpropagation(X, loss_gradient)

        return loss


