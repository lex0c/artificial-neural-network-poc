import math
import json


# It is an activation function that resets all negative values and keeps all positive values as they are. 
# It is especially useful in deep neural networks as it helps mitigate the gradient vanishing problem.
def relu(x):
    return max(0, x)


# It is an activation function that compresses the output between 0 and 1. It is useful in output layers of 
# binary classification problems.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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

# Min-Max Scaling is a normalization technique that transforms features by scaling each feature to a 
# specific range, usually [0, 1] or [-1, 1].
def normalize_minmax(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def one_hot_encode(unique_elements, element):
    if element not in unique_elements:
        raise ValueError(f"{element} not found in {unique_elements}")

    encoding = [0] * len(unique_elements)
    encoding[unique_elements.index(element)] = 1

    return encoding


def save_model(model_name, data):
    path = 'models/' + model_name + '.json'
    with open(path, 'w') as file:
        json.dump(data, file, indent=2)


def load_model(model_name):
    path = 'models/' + model_name + '.json'
    with open(path, 'r') as file:
        return json.load(file)


