import numpy as np


# Activation functions are applied to the weighted sum z of each neuron. Without them a stack of layers would still
# compute a linear function of its input: the non-linearity is what lets a network learn curved relationships.
#
# The derivatives are evaluated at z (the pre-activation), as the chain rule needs (see activation_backward).


# ReLU: max(0, z). Cheap, and its gradient doesn't fade for positive z, which helps deeper networks learn.
def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(np.asarray(z) > 0, 1, 0)


# Sigmoid: squashes z into (0, 1), so its output can be read as a probability (e.g. a binary classifier).
def sigmoid(z):
    z = np.array(z)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# Linear: returns z unchanged. Used in the output layer when the result can be any number (regression, Q-values).
def linear(z):
    return z


def linear_derivative(z):
    return np.ones_like(z)


# Softmax: turns the outputs of a layer into probabilities that sum to 1, one per class (multi-class classification).
# It normalizes along the last axis, so a batch with shape (batch_size, num_classes) gets one distribution per sample.
# Subtracting the max doesn't change the result (it cancels out in the division), but keeps exp from overflowing.
def softmax(z):
    z = np.asarray(z, dtype=float)
    exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


ACTIVATIONS = {"relu": relu, "sigmoid": sigmoid, "linear": linear, "softmax": softmax}

# Softmax isn't here: each of its outputs depends on every z of the sample, see activation_backward
ELEMENTWISE_DERIVATIVES = {"relu": relu_derivative, "sigmoid": sigmoid_derivative, "linear": linear_derivative}


def activation_fn(name, z):
    if name not in ACTIVATIONS:
        raise ValueError(f"Invalid activation function: {name}. Use one of {list(ACTIVATIONS)}")

    return ACTIVATIONS[name](z)


# Chain rule through the activation: from the gradient of the loss with respect to the output y = f(z) of the neurons
# (grad_output) to the gradient with respect to z.
#
# z, output and grad_output have shape (num_neurons,) for one sample or (batch_size, num_neurons) for a batch.
def activation_backward(name, z, output, grad_output):
    if name == "softmax":
        # Every softmax output s depends on every z of its sample, through the Jacobian J = diag(s) - s s^T.
        # Multiplying grad_output by it simplifies to s * (grad_output - sum(grad_output * s)).
        return output * (grad_output - np.sum(grad_output * output, axis=-1, keepdims=True))

    # Each output depends only on its own z: dL/dz = dL/dy * f'(z)
    return grad_output * ELEMENTWISE_DERIVATIVES[name](z)
