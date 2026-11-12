import matplotlib.pyplot as plt
import numpy as np

from cli import display_path, project_path, training_args
from data import set_seed
from feedforward import FeedForward
from spirals import make_spirals


# Overfitting and regularization, in pictures: a large network (two hidden layers of 64 neurons) learns two spirals
# from only 100 noisy points of each. Without regularization it bends its boundary around every point, even the ones
# the noise pushed into the other spiral, and gets much worse on new points than on the ones it learned. The same
# network is trained with L2, dropout and early stopping, and once with too much regularization, which can't follow
# the spirals anymore (underfitting). regularization.png shows the boundary of each one.

args = training_args("Overfitting on few noisy points, and what L2, dropout and early stopping do about it.",
                     dashboard=False, epochs=1000)  # five trainings one after the other: nothing to follow live
set_seed(args.seed)

points, labels = make_spirals(num_samples=100, noise=0.1)          # to train
val_points, val_labels = make_spirals(num_samples=100, noise=0.1)  # to choose the epoch, for early stopping
test_points, test_labels = make_spirals(num_samples=1000, noise=0.1)

TRAININGS = [  # name, options of add_layer, options of train
    ("No regularization", {}, {}),
    ("L2 0.0003", {}, {"l2_lambda": 0.0003}),
    ("Dropout 0.2", {"dropout": 0.2}, {}),
    ("Early stopping (patience 50)", {}, {"validation_data": (val_points, val_labels), "patience": 50}),
    ("Too much: L2 0.003", {}, {"l2_lambda": 0.003}),
]

grid_x, grid_y = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
panels = []

for name, layer_options, train_options in TRAININGS:
    model = FeedForward()
    model.add_layer(num_inputs=2, num_neurons=64, act_fn='tanh')
    model.add_layer(num_inputs=64, num_neurons=64, act_fn='tanh', **layer_options)
    model.add_layer(num_inputs=64, num_neurons=1, act_fn='sigmoid', **layer_options)
    model.configure(loss='binary_crossentropy', optimizer='adam')
    history = model.train(points, labels, epochs=args.epochs, learning_rate=0.01, batch_size=16, **train_options)

    _, train_accuracy = model.evaluate(points, labels)
    _, test_accuracy = model.evaluate(test_points, test_labels)
    probabilities = model.predict(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    panels.append((name, train_accuracy, test_accuracy, probabilities))
    epochs = f", stopped after {len(history['loss'])} epochs" if "best_epoch" in history else ""
    print(f"{name:30s} right on the training points {train_accuracy * 100:5.1f}%, on new points {test_accuracy * 100:5.1f}%{epochs}")


figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.6))

for axis, (name, train_accuracy, test_accuracy, probabilities) in zip(axes, panels):
    axis.contourf(grid_x, grid_y, probabilities, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.6)
    axis.contour(grid_x, grid_y, probabilities, levels=[0.5], colors="black", linewidths=1.5)
    axis.scatter(*points[labels == 0].T, s=8, c="tab:blue", edgecolors="white", linewidths=0.3)
    axis.scatter(*points[labels == 1].T, s=8, c="tab:red", edgecolors="white", linewidths=0.3)
    axis.set_title(f"{name}\ntraining points {train_accuracy * 100:.0f}%, new points {test_accuracy * 100:.0f}%")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")

figure.suptitle("The same network on 100 noisy points per spiral: the gap between the points it learned and new points is the overfitting")
plt.tight_layout(rect=(0, 0, 1, 0.9))
figure_filepath = project_path('regularization.png')
plt.savefig(figure_filepath, dpi=120)
print(f"\nDecision boundaries saved to {display_path(figure_filepath)}")
