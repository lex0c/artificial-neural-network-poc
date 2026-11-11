import math
import random
import numpy as np


# It is an activation function that resets all negative values and keeps all positive values as they are. 
# It is especially useful in deep neural networks as it helps mitigate the gradient vanishing problem.
def relu(x):
    return np.maximum(0, x)


# Returns the derivative of the ReLU function evaluated at x.
def relu_derivative(x):
    return np.where(np.asarray(x) > 0, 1, 0)


# It is an activation function that compresses the output between 0 and 1. It is useful in output layers of 
# binary classification problems.
def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))


# Returns the derivative of the sigmoid function evaluated at x.
def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


# The Softmax function converts a vector of numbers into a vector of probabilities, 
# where each element represents the probability of the input belonging to one of the classes. 
# It is useful in the output layer of multi-class classification problems.
#
# It normalizes along the last axis, so a batch with shape (batch_size, num_classes) gets one
# probability distribution per sample.
def softmax(x):
    x = np.asarray(x, dtype=float)
    max_x = np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x - max_x)
    return exps / np.sum(exps, axis=-1, keepdims=True)


# Returns the derivative (Jacobian matrix) of the softmax function evaluated at x, for a single sample.
def softmax_derivative(x):
    s = softmax(x).reshape(-1, 1)
    jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
    return jacobian_matrix


# A linear function does not change the input, that is, the output is the same as the input. 
# It is useful when you want the output of the neural network to be a continuous value (regression).
def linear(x):
    return x


# Returns the derivative of the linear function, which is 1.
def linear_derivative(x):
    return np.ones_like(x)


def activation_fn(fn, z):
    if fn == "relu":
        return relu(z)
    elif fn == "sigmoid":
        return sigmoid(z)
    elif fn == "linear":
        return linear(z)
    elif fn == "softmax":
        return softmax(z)

    raise Exception("Invalid activation function.")


def activation_derivative_fn(fn, z):
    if fn == "relu":
        return relu_derivative(z)
    elif fn == "sigmoid":
        return sigmoid_derivative(z)
    elif fn == "linear":
        return linear_derivative(z)
    elif fn == "softmax":
        return softmax_derivative(z)

    raise Exception("Invalid derivative activation function.")


# Backpropagates the gradient of the loss in relation to the activation output (grad) through the
# activation function, returning the gradient in relation to the pre-activation z.
#
# z, output and grad have shape (num_neurons,) for a single sample or (batch_size, num_neurons) for a batch.
def activation_backward(fn, z, output, grad):
    if fn == "softmax":
        # Every softmax output depends on every z of the same sample, so the gradient goes through the
        # Jacobian J = diag(s) - s s^T (symmetric): J grad = s * (grad - sum(grad * s))
        return output * (grad - np.sum(grad * output, axis=-1, keepdims=True))

    # Element-wise activations: each output depends only on its own z
    return grad * activation_derivative_fn(fn, z)


# Min-Max Scaling is a normalization technique that transforms features by scaling each feature to a
# specific range, usually [0, 1] or [-1, 1].
def normalize_minmax(data):
    return apply_minmax(data, *minmax_stats(data))


# Returns the minimum and maximum of each feature (column), to scale other data the same way later.
def minmax_stats(data):
    data = np.asarray(data, dtype=float)
    return np.min(data, axis=0), np.max(data, axis=0)


# Scales each feature to [0, 1] using the given minimum and maximum. Values outside the original range
# (e.g. new data) land outside [0, 1], which is expected.
def apply_minmax(data, min_vals, max_vals):
    value_range = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)  # a constant feature would divide by 0
    return (np.asarray(data, dtype=float) - min_vals) / value_range


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


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')

    if not num_classes:
        num_classes = np.max(y) + 1

    categorical = np.zeros((y.shape[0], num_classes))
    categorical[np.arange(y.shape[0]), y] = 1

    return categorical


def to_categorical_int(y_categorical):
    y_categorical = np.array(y_categorical)
    return np.argmax(y_categorical, axis=1)


# All loss functions accept a single sample (y_pred with shape (num_outputs,)) or a batch
# (y_pred with shape (batch_size, num_outputs)) and return the mean loss over the samples.
# Their derivatives are the gradient of that mean, so they already include the 1/batch_size factor.
def num_samples(y_pred):
    return y_pred.shape[0] if y_pred.ndim > 1 else 1


# The MSE measures the average of the squares of the differences between the predicted values and the actual values.
# It provides a simple and effective means of assessing the performance of a model.
def mse_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_pred - y_true) ** 2)


# Returns the derivative of the mse_loss function.
def mse_loss_derivative(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 2 * (y_pred - y_true) / y_pred.size


# Sparse Categorical Crossentropy computes the loss between the true labels and the predicted probabilities output.
# The true labels in this case are integers that represent the class indices directly, and the predicted probabilities
# are typically the output of a softmax layer which converts logits to probabilities that sum to one.
#
# y_true can be a single class index, a list of class indices or a column of class indices (shape (batch_size, 1)).
def sparse_categorical_crossentropy(y_true, y_pred):
    y_true = np.array(y_true, dtype=int).reshape(-1)
    y_pred = np.atleast_2d(np.array(y_pred, dtype=float))

    # Ensure predictions are clipped to prevent log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Select the probability of the true class for each sample
    true_probs = y_pred[np.arange(len(y_true)), y_true]

    # Calculate the average loss
    return -np.mean(np.log(true_probs))


# Returns the derivative of the sparse_categorical_crossentropy function.
def sparse_categorical_crossentropy_derivative(y_true, y_pred):
    y_true = np.array(y_true, dtype=int).reshape(-1)
    y_pred = np.array(y_pred, dtype=float)

    # Ensure predictions are clipped to prevent division by 0
    probs = np.clip(np.atleast_2d(y_pred), 1e-7, 1 - 1e-7)
    rows = np.arange(len(y_true))

    # Only the probability of the true class affects the loss of each sample
    gradients = np.zeros_like(probs)
    gradients[rows, y_true] = -1 / probs[rows, y_true] / len(y_true)

    return gradients.reshape(y_pred.shape)


def categorical_crossentropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clip predictions to avoid log(0)
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=-1))


def categorical_crossentropy_derivative(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clip predictions to avoid division by 0
    return -y_true / y_pred / num_samples(y_pred)


# For a softmax output trained with cross-entropy, the chain rule through the softmax simplifies to
# (y_pred - y_true) / batch_size in relation to the pre-activation z. Using it directly avoids dividing by
# tiny probabilities, whose gradient would vanish when the softmax saturates (e.g. with unscaled inputs).
#
# The simplification assumes each row of y_true sums to 1 (one-hot or soft labels), as class probabilities do.
def softmax_crossentropy_delta(fn, y_true, y_pred):
    y_pred = np.array(y_pred, dtype=float)

    if fn == "sparse_categorical_crossentropy":
        class_indices = np.array(y_true, dtype=int).reshape(-1)
        y_true = to_categorical(class_indices, num_classes=y_pred.shape[-1]).reshape(y_pred.shape)

    return (y_pred - np.array(y_true)) / num_samples(y_pred)


def is_softmax_crossentropy(act_fn, loss):
    return act_fn == "softmax" and loss in ("categorical_crossentropy", "sparse_categorical_crossentropy")


def loss_fn(fn, y_true, y_pred):
    if fn == "mse":
        return mse_loss(y_true, y_pred)
    elif fn == "sparse_categorical_crossentropy":
        return sparse_categorical_crossentropy(y_true, y_pred)
    elif fn == "categorical_crossentropy":
        return categorical_crossentropy(y_true, y_pred)

    raise Exception("Invalid loss function.")


def loss_derivative_fn(fn, y_true, y_pred):
    if fn == "mse":
        return mse_loss_derivative(y_true, y_pred)
    elif fn == "sparse_categorical_crossentropy":
        return sparse_categorical_crossentropy_derivative(y_true, y_pred)
    elif fn == "categorical_crossentropy":
        return categorical_crossentropy_derivative(y_true, y_pred)

    raise Exception("Invalid derivative loss function.")


# Gradient clipping technique involves clipping the gradients during backpropagation to ensure they do not exceed 
# a certain threshold, thus maintaining stability.
def clip_gradients(gradients, threshold=5.0):
    original_gradients = np.copy(gradients)
    clipped_gradients = np.clip(gradients, -threshold, threshold)

    # Check if clipping has taken place
    if np.any(clipped_gradients != original_gradients):
        clipping_locations = np.where(clipped_gradients != original_gradients)
        #print("Clipped indices:", clipping_locations)
        #print("Original gradients at clipped positions:", original_gradients[clipping_locations])
        #print("Clipped gradients at clipped positions:", clipped_gradients[clipping_locations])

    return clipped_gradients


# Normalizing gradients involves adjusting the scale of the gradients based on their norm before applying the weight 
# update. This is done by dividing the gradients by their norm (or by some function of the norm). The aim is to ensure 
# that the magnitude of the gradients is neither too large nor too small, while maintaining the direction of the gradients.
def normalize_gradients(gradients):
    norm = np.linalg.norm(gradients, ord=2)

    if norm > 0:
        gradients = gradients / norm

    return gradients


# Seeds the random number generators used in training (NumPy, Python's random and, optionally, a Gymnasium
# environment and its action space), so a run can be reproduced. With seed=None every run is different.
def set_seed(seed, env=None):
    np.random.seed(seed)
    random.seed(seed)

    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)


# Spliting a dataset into smaller chunks.
def create_batches(inputs, targets, batch_size):
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i + batch_size], targets[i:i + batch_size]


