import random

import numpy as np


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
    data = np.asarray(data, dtype=float)
    return np.min(data, axis=0), np.max(data, axis=0)


# Values outside the training range (e.g. new data) land outside [0, 1], which is expected.
def apply_minmax(data, min_vals, max_vals):
    value_range = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)  # a constant feature would divide by 0
    return (np.asarray(data, dtype=float) - min_vals) / value_range


# Splits a dataset into consecutive batches of batch_size samples (the last one is smaller if the size doesn't
# divide evenly).
def create_batches(inputs, targets, batch_size):
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i + batch_size], targets[i:i + batch_size]


# Seeds the random number generators used in training (NumPy, Python's random and, optionally, a Gymnasium
# environment and its action space), so a run can be reproduced. With seed=None every run is different.
def set_seed(seed, env=None):
    np.random.seed(seed)
    random.seed(seed)

    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
