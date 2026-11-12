import random

import numpy as np


# Floating-point values as an array that keeps their precision: float32 stays float32 (ssnn/ computes in float32), and
# anything else (a list, integers) becomes float64. It doesn't copy an array that is already floating-point, so the
# functions that use it must not change it in place.
def as_float(values):
    values = np.asarray(values)
    return values if np.issubdtype(values.dtype, np.floating) else values.astype(float)


# One-hot encoding: class index 2 out of 4 classes becomes [0, 0, 1, 0]. categorical_crossentropy compares this
# vector with the probabilities predicted by a softmax output.
def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')

    if not num_classes:
        num_classes = np.max(y) + 1

    categorical = np.zeros((y.shape[0], num_classes))
    categorical[np.arange(y.shape[0]), y] = 1

    return categorical


# Min-max scaling maps each feature (column) to [0, 1] with (x - min) / (max - min). Features with very different
# scales (e.g. a temperature around 20 and a humidity around 80) would otherwise get very different gradients.
#
# minmax_stats returns the minimum and maximum of each feature of the training data, so new data can later be
# scaled with the same values by apply_minmax.
def minmax_stats(data):
    data = as_float(data)
    return np.min(data, axis=0), np.max(data, axis=0)


# Values outside the training range (e.g. new data) land outside [0, 1], which is expected.
def apply_minmax(data, min_vals, max_vals):
    value_range = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)  # a constant feature would divide by 0
    return (as_float(data) - min_vals) / value_range


# Splits a sequence (in train(), the order of the samples in an epoch) into consecutive batches of batch_size items.
# The last one is smaller if the size doesn't divide evenly.
def create_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# Seeds the random number generators used in training (NumPy, Python's random and, optionally, a Gymnasium
# environment and its action space), so a run can be reproduced. With seed=None every run is different.
def set_seed(seed, env=None):
    np.random.seed(seed)
    random.seed(seed)

    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
