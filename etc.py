import math
import json
import random
import numpy as np


# It is an activation function that resets all negative values and keeps all positive values as they are. 
# It is especially useful in deep neural networks as it helps mitigate the gradient vanishing problem.
def relu(x):
    return np.maximum(0, x)


# Returns the derivative of the ReLU function evaluated at x.
def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# It is an activation function that compresses the output between 0 and 1. It is useful in output layers of 
# binary classification problems.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Returns the derivative of the sigmoid function evaluated at x.
def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


# The Softmax function converts a vector of numbers into a vector of probabilities, 
# where each element represents the probability of the input belonging to one of the classes. 
# It is useful in the output layer of multi-class classification problems.
def softmax(x):
    exps = [math.exp(i) for i in x]
    sum_exps = sum(exps)
    return [j/sum_exps for j in exps]


# A linear function does not change the input, that is, the output is the same as the input. 
# It is useful when you want the output of the neural network to be a continuous value (regression).
def linear(x):
    return x


# Returns the derivative of the linear function, which is 1.
def linear_derivative(x):
    return 1


# Add other activation functions as needed
def activation_derivative(act_fn, z):
    if act_fn == "relu":
        return relu_derivative(z)
    elif act_fn == "sigmoid":
        return sigmoid_derivative(z)
    elif act_fn == "linear":
        return linear_derivative(z)


# Min-Max Scaling is a normalization technique that transforms features by scaling each feature to a 
# specific range, usually [0, 1] or [-1, 1].
def normalize_minmax(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


# This function is responsible for converting an element into a "one-hot encode" representation. 
# In "one-hot encoding", each single element is represented by a vector where one of the elements is 
# 1 and the rest are 0. For example, for three classes A, B, and C, A would be [1,0,0] , B would be [0,1,0], 
# and C would be [0,0,1].
def one_hot_encode(unique_elements, element):
    if element not in unique_elements:
        raise ValueError(f"{element} not found in {unique_elements}")

    encoding = [0] * len(unique_elements)
    encoding[unique_elements.index(element)] = 1

    return encoding


# This function represents an epsilon-greedy strategy applied to softmax probabilities. 
# It takes as input a list of softmax probabilities, softmax_probs, and an epsilon value that controls 
# the ratio between exploration and exploitation.
def epsilon_greedy_softmax(softmax_probs, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:  # Explore: Choose a random action.
        return random.randint(0, len(softmax_probs) - 1)
    else:  # Exploit: Choose action based on softmax probabilities.
        return max(range(len(softmax_probs)), key=lambda i: softmax_probs[i])


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def mse_loss_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_true)


def save_model(model_name, data):
    path = 'models/' + model_name + '.json'
    with open(path, 'w') as file:
        json.dump(data, file, indent=2)


def load_model(model_name):
    path = 'models/' + model_name + '.json'
    with open(path, 'r') as file:
        return json.load(file)


