import numpy as np

from feedforward import FeedForward


# Dense baselines for the quality of the experiments: an MLP with one hidden layer,
# with a size chosen to match the SSNN in total parameters (the same stored capacity) or in parameters used per event
# (about the same work per inference). They use the FeedForward of this repository, which computes in float64 and
# limits BLAS to one thread: fine to compare accuracy, not to measure latency (the benchmark of experiment C has its
# own baselines).


# Parameters of an MLP with one hidden layer: (d + 1) * hidden + (hidden + 1) * C
def mlp_parameters(num_inputs, num_hidden, num_classes):
    return (num_inputs + 1) * num_hidden + (num_hidden + 1) * num_classes


# The hidden size whose parameter count is closest to target
def hidden_for_parameters(target, num_inputs, num_classes):
    return max(1, round((target - num_classes) / (num_inputs + 1 + num_classes)))


def dense_mlp(num_inputs, num_hidden, num_classes, seed):
    np.random.seed(seed)  # FeedForward draws its weights and shuffles with NumPy's global generator
    model = FeedForward()
    model.add_layer(num_inputs=num_inputs, num_neurons=num_hidden, act_fn='relu')
    model.add_layer(num_inputs=num_hidden, num_neurons=num_classes, act_fn='softmax')
    model.configure(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


# Trains with the same epochs, minibatch size and learning rate as the SSNN, and keeps the epoch with the lowest
# validation loss like ssnn.training.train does (patience equal to the epochs: it never stops early, but restores the
# best epoch)
def train_mlp(model, train_set, val_set, config):
    return model.train(train_set[0], train_set[1], epochs=config["epochs"], learning_rate=config["learning_rate"],
                       batch_size=config["batch_size"], validation_data=(val_set[0], val_set[1]), patience=config["epochs"])
