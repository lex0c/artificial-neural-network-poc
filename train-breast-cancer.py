import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

from cli import display_path, keep_dashboard_open, project_path, start_dashboard, training_args
from data import set_seed
from feedforward import FeedForward


# Binary classification on real data: is a breast tumor malignant? Each of the 569 samples of the breast cancer
# dataset of scikit-learn (no download) has 30 measurements of the cell nuclei in an image of a biopsy, such as their
# radius, texture and smoothness. With so few samples a network overfits quickly, so three trainings of the same
# network are compared: without regularization, with early stopping and with dropout. breast-cancer.png shows their
# validation loss along the training.

args = training_args("Is a breast tumor malignant? Compares training without regularization, with early stopping "
                     "and with dropout.", epochs=500)
set_seed(args.seed)

data = load_breast_cancer()
features, malignant = data.data, 1 - data.target  # the dataset uses 0 for malignant: here 1 means malignant

# 60% to train, 20% to validate (choosing the epoch, for early stopping) and 20% to test, in a random order
order = np.random.permutation(len(features))
test, val, train = order[:114], order[114:228], order[228:]


# The probability that the tumor is malignant: one sigmoid output with binary cross-entropy
def build(dropout=0.0):
    model = FeedForward()
    model.add_layer(num_inputs=30, num_neurons=64, act_fn='relu')
    model.add_layer(num_inputs=64, num_neurons=1, act_fn='sigmoid', dropout=dropout)
    model.configure(loss='binary_crossentropy', optimizer='adam')
    model.fit_normalization(features[train])  # the measurements have very different scales (an area of 1000, a smoothness of 0.1)
    return model


def plural(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


TRAININGS = [  # name, model, patience
    ("no regularization", build(), None),
    ("early stopping (patience 20)", build(), 20),
    ("dropout 0.5", build(dropout=0.5), None),
]
monitor = start_dashboard(args.dashboard, {"early stopping": TRAININGS[1][1]})

print(f"{len(train)} tumors to train, {len(val)} to validate and {len(test)} to test, "
      f"{int(malignant[test].sum())} of them malignant\n")
histories = []

for name, model, patience in TRAININGS:
    history = model.train(features[train], malignant[train], epochs=args.epochs, learning_rate=0.001, batch_size=16,
                          validation_data=(features[val], malignant[val]), patience=patience)
    histories.append(history)

    test_loss, test_accuracy = model.evaluate(features[test], malignant[test])
    predicted = model.predict(features[test])[:, 0] > 0.5
    missed = int(np.sum(~predicted & (malignant[test] == 1)))  # malignant tumors taken as benign
    false_alarms = int(np.sum(predicted & (malignant[test] == 0)))
    epochs = f"{len(history['loss'])} epochs" + (f", weights of epoch {history['best_epoch']}" if "best_epoch" in history else "")

    print(f"{name}: {epochs}")
    print(f"    test accuracy {test_accuracy * 100:.1f}%, test loss {test_loss:.3f}, "
          f"{plural(missed, 'malignant tumor')} missed, {plural(false_alarms, 'false alarm')}")


figure, axis = plt.subplots(figsize=(9, 5))

for (name, _, _), history in zip(TRAININGS, histories):
    line, = axis.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label=f"{name}: validation")
    if "best_epoch" in history:
        axis.axvline(history["best_epoch"], color=line.get_color(), linestyle=":", label="the epoch early stopping keeps")

axis.plot(range(1, len(histories[0]["loss"]) + 1), histories[0]["loss"], color="gray", linestyle="--",
          label="no regularization: training")
axis.set_ylim(0, 0.5)
axis.set_xlabel("epoch")
axis.set_ylabel("loss (binary cross-entropy)")
axis.set_title("The training loss keeps falling, the validation loss rises again: overfitting")
axis.legend()

plt.tight_layout()
figure_filepath = project_path('breast-cancer.png')
plt.savefig(figure_filepath, dpi=120)
print(f"\nValidation curves saved to {display_path(figure_filepath)}")
keep_dashboard_open(monitor)
