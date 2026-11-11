import numpy as np

from data import to_categorical


# A loss function measures how far the predictions are from the targets, as a single number that training minimizes.
#
# Each loss takes a single sample (y_pred with shape (num_outputs,)) or a batch (shape (batch_size, num_outputs)) and
# returns the mean over the samples. Each derivative is the gradient of that mean with respect to y_pred, so it already
# includes the division by the number of samples (for mse, by the number of samples times the number of outputs).


def num_samples(y_pred):
    return y_pred.shape[0] if y_pred.ndim > 1 else 1


# Mean squared error: the mean of (y_pred - y_true)^2. Used for regression; squaring makes large errors weigh more.
def mse_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_pred - y_true) ** 2)


def mse_loss_derivative(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 2 * (y_pred - y_true) / y_pred.size


# Categorical cross-entropy: -sum(y_true * log(y_pred)) per sample. With one-hot labels this is minus the log of the
# probability given to the true class: 0 when it is 1, and very large when a confident prediction is wrong.
# Used for classification with a softmax output. Probabilities are clipped away from 0, so log(0) can't happen.
def categorical_crossentropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=-1))


def categorical_crossentropy_derivative(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    return -y_true / y_pred / num_samples(y_pred)


# Sparse categorical cross-entropy: the same loss, with the labels given as class indices (2 instead of [0, 0, 1]).
# y_true can be a single index, a list of indices or a column of indices (shape (batch_size, 1)).
def sparse_categorical_crossentropy(y_true, y_pred):
    y_true = np.array(y_true, dtype=int).reshape(-1)
    y_pred = np.clip(np.atleast_2d(np.array(y_pred, dtype=float)), 1e-7, 1 - 1e-7)

    true_probs = y_pred[np.arange(len(y_true)), y_true]  # the probability given to the true class of each sample
    return -np.mean(np.log(true_probs))


def sparse_categorical_crossentropy_derivative(y_true, y_pred):
    y_true = np.array(y_true, dtype=int).reshape(-1)
    y_pred = np.array(y_pred, dtype=float)
    probs = np.clip(np.atleast_2d(y_pred), 1e-7, 1 - 1e-7)
    rows = np.arange(len(y_true))

    # Only the probability of the true class appears in the loss of each sample
    gradients = np.zeros_like(probs)
    gradients[rows, y_true] = -1 / probs[rows, y_true] / len(y_true)

    return gradients.reshape(y_pred.shape)


LOSSES = {
    "mse": (mse_loss, mse_loss_derivative),
    "categorical_crossentropy": (categorical_crossentropy, categorical_crossentropy_derivative),
    "sparse_categorical_crossentropy": (sparse_categorical_crossentropy, sparse_categorical_crossentropy_derivative),
}


def check_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Invalid loss function: {name}. Use one of {list(LOSSES)}")


def loss_fn(name, y_true, y_pred):
    check_loss(name)
    return LOSSES[name][0](y_true, y_pred)


def loss_derivative_fn(name, y_true, y_pred):
    check_loss(name)
    return LOSSES[name][1](y_true, y_pred)


# Softmax with cross-entropy: combining the derivative of the loss with the softmax Jacobian, the gradient with
# respect to z of the output layer simplifies to (y_pred - y_true) / batch_size. Using it directly avoids dividing by
# tiny probabilities, whose gradient would vanish when the softmax saturates (outputs close to 0 or 1).
#
# The simplification assumes each row of y_true sums to 1 (one-hot or soft labels), as class probabilities do.
def softmax_crossentropy_delta(name, y_true, y_pred):
    y_pred = np.array(y_pred, dtype=float)

    if name == "sparse_categorical_crossentropy":
        class_indices = np.array(y_true, dtype=int).reshape(-1)
        y_true = to_categorical(class_indices, num_classes=y_pred.shape[-1]).reshape(y_pred.shape)

    return (y_pred - np.array(y_true)) / num_samples(y_pred)


def is_softmax_crossentropy(act_fn, loss):
    return act_fn == "softmax" and loss in ("categorical_crossentropy", "sparse_categorical_crossentropy")


# Gradient clipping limits each value of a gradient to [-threshold, threshold], so a single huge error (an outlier,
# or an early Q-value target far from the prediction) can't produce an update large enough to make training diverge.
def clip_gradients(gradients, threshold):
    return np.clip(gradients, -threshold, threshold)
