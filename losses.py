import numpy as np

from data import as_float, to_categorical


# A loss function measures how far the predictions are from the targets, as a single number that training minimizes.
#
# Each loss takes a single sample (y_pred with shape (num_outputs,)) or a batch (shape (batch_size, num_outputs)) and
# returns the mean over the samples. Each derivative is the gradient of that mean with respect to y_pred, so it already
# includes the division by the number of samples (for mse, by the number of samples times the number of outputs).
#
# The cross-entropy losses clip the probabilities to [1e-7, 1 - 1e-7] to avoid log(0). The derivatives keep the formula
# of the unclipped loss: where a probability is clipped, the clipped loss would be flat and its gradient 0, which would
# stop the model from learning exactly where it is most wrong.


def num_samples(y_pred):
    return y_pred.shape[0] if y_pred.ndim > 1 else 1


# The predictions keep their precision (see as_float in data.py) and the targets take the same one, so float32
# predictions give a float32 loss and float32 gradients (float32 minus int64 targets would otherwise be float64)
def as_arrays(y_true, y_pred):
    y_pred = as_float(y_pred)
    return np.asarray(y_true).astype(y_pred.dtype, copy=False), y_pred


# Mean squared error: the mean of (y_pred - y_true)^2. Used for regression; squaring makes large errors weigh more.
def mse_loss(y_true, y_pred):
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean((y_pred - y_true) ** 2)


def mse_loss_derivative(y_true, y_pred):
    y_true, y_pred = as_arrays(y_true, y_pred)
    return 2 * (y_pred - y_true) / y_pred.size


# Categorical cross-entropy: -sum(y_true * log(y_pred)) per sample. With one-hot labels this is minus the log of the
# probability given to the true class: close to 0 when it is close to 1, and very large when a confident prediction is
# wrong.
# Used for classification with a softmax output. Probabilities are clipped away from 0, so log(0) can't happen.
def categorical_crossentropy(y_true, y_pred):
    y_true, y_pred = as_arrays(y_true, y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=-1))


def categorical_crossentropy_derivative(y_true, y_pred):
    y_true, y_pred = as_arrays(y_true, y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -y_true / y_pred / num_samples(y_pred)


# Sparse categorical cross-entropy: the same loss, with the labels given as class indices (2 instead of [0, 0, 1]).
# y_true can be a single index, a list of indices or a column of indices (shape (batch_size, 1)).
def sparse_categorical_crossentropy(y_true, y_pred):
    y_true = np.array(y_true, dtype=int).reshape(-1)
    y_pred = np.clip(np.atleast_2d(as_float(y_pred)), 1e-7, 1 - 1e-7)

    true_probs = y_pred[np.arange(len(y_true)), y_true]  # the probability given to the true class of each sample
    return -np.mean(np.log(true_probs))


def sparse_categorical_crossentropy_derivative(y_true, y_pred):
    y_true = np.array(y_true, dtype=int).reshape(-1)
    y_pred = as_float(y_pred)
    probs = np.clip(np.atleast_2d(y_pred), 1e-7, 1 - 1e-7)
    rows = np.arange(len(y_true))

    # Only the probability of the true class appears in the loss of each sample
    gradients = np.zeros_like(probs)
    gradients[rows, y_true] = -1 / probs[rows, y_true] / len(y_true)

    return gradients.reshape(y_pred.shape)


# Binary cross-entropy: -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)) for each output, averaged. Each output
# is a yes/no answer given as the probability of "yes" (a sigmoid output), and y_true is 1 for yes and 0 for no. Unlike
# the categorical cross-entropy, it has a term for "no" as well, so it also pushes the probability down when the answer
# is no. Several sigmoid outputs are several independent yes/no answers (e.g. tags of an image, any number of them true).
def binary_crossentropy(y_true, y_pred):
    y_true, y_pred = as_arrays(y_true, y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def binary_crossentropy_derivative(y_true, y_pred):
    y_true, y_pred = as_arrays(y_true, y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_pred.size  # -y/p + (1 - y)/(1 - p), averaged


LOSSES = {
    "mse": (mse_loss, mse_loss_derivative),
    "categorical_crossentropy": (categorical_crossentropy, categorical_crossentropy_derivative),
    "sparse_categorical_crossentropy": (sparse_categorical_crossentropy, sparse_categorical_crossentropy_derivative),
    "binary_crossentropy": (binary_crossentropy, binary_crossentropy_derivative),
}


def check_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Invalid loss function: {name}. Use one of {list(LOSSES)}")


# The cross-entropy losses read the outputs as probabilities, so each one needs its output activation: the categorical
# ones a softmax (a distribution over the classes) and the binary one a sigmoid (a probability per yes/no answer). With
# another output they would train wrong without any error. For example, the categorical cross-entropy of a single
# sigmoid output is -y * log(p): it is 0 for every sample with y = 0, so nothing pushes p down, and the model learns to
# answer 1 for everything.
OUTPUT_OF_LOSS = {
    "categorical_crossentropy": ("softmax", "for yes/no answers with sigmoid outputs, use binary_crossentropy"),
    "sparse_categorical_crossentropy": ("softmax", "for yes/no answers with sigmoid outputs, use binary_crossentropy"),
    "binary_crossentropy": ("sigmoid", "for one class out of several with a softmax output, use categorical_crossentropy"),
}


def check_output_for_loss(act_fn, loss):
    if loss in OUTPUT_OF_LOSS and act_fn != OUTPUT_OF_LOSS[loss][0]:
        needed, hint = OUTPUT_OF_LOSS[loss]
        raise ValueError(f"{loss} needs a {needed} output layer, got {act_fn} ({hint})")


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
    y_pred = as_float(y_pred)

    if name == "sparse_categorical_crossentropy":
        class_indices = np.array(y_true, dtype=int).reshape(-1)
        y_true = to_categorical(class_indices, num_classes=y_pred.shape[-1]).reshape(y_pred.shape)

    y_true, y_pred = as_arrays(y_true, y_pred)
    return (y_pred - y_true) / num_samples(y_pred)


def is_softmax_crossentropy(act_fn, loss):
    return act_fn == "softmax" and loss in ("categorical_crossentropy", "sparse_categorical_crossentropy")


# Sigmoid with binary cross-entropy: the same simplification. The derivative of the loss, (y_pred - y_true) /
# (y_pred * (1 - y_pred)), times the derivative of the sigmoid, y_pred * (1 - y_pred), leaves (y_pred - y_true), divided
# by the number of values averaged. Without it, a confident wrong answer (y_pred close to 0 when y_true is 1) would
# divide by almost 0 and multiply by almost 0.
def sigmoid_binary_crossentropy_delta(y_true, y_pred):
    y_true, y_pred = as_arrays(y_true, y_pred)
    return (y_pred - y_true.reshape(y_pred.shape)) / y_pred.size


def is_sigmoid_binary_crossentropy(act_fn, loss):
    return act_fn == "sigmoid" and loss == "binary_crossentropy"


# Gradient clipping limits each value of a gradient to [-threshold, threshold], so a single huge error (an outlier,
# or an early Q-value target far from the prediction) can't produce an update large enough to make training diverge.
# It is applied to dL/dy_pred, which already includes the division by the batch size (and, for mse, by the number of
# outputs): with large batches, an error has to be much larger to reach the threshold.
def clip_gradients(gradients, threshold):
    return np.clip(gradients, -threshold, threshold)
