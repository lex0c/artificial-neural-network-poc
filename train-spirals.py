import matplotlib.pyplot as plt
import numpy as np

from cli import display_path, keep_dashboard_open, project_path, start_dashboard, training_args
from data import set_seed
from feedforward import FeedForward
from spirals import make_spirals


# Decision boundaries: two spirals wind around each other, one per class (spirals.py), and two models learn to tell
# them apart. A model without hidden layers can only draw a straight line between the classes; a network with two
# hidden layers bends its boundary to follow the arms. spirals.png shows both boundaries, and how the network's one
# takes shape during the training.

args = training_args("Decision boundaries: a straight line against a network with hidden layers, on two spirals.", epochs=300)
set_seed(args.seed)

points, labels = make_spirals()
test_points, test_labels = make_spirals()  # new points from the same spirals


# A yes/no answer (is the point on spiral 1?): one sigmoid output with binary cross-entropy
def build(hidden_layers):
    model = FeedForward()
    num_inputs = 2  # the coordinates of the point

    for num_neurons in hidden_layers:
        model.add_layer(num_inputs=num_inputs, num_neurons=num_neurons, act_fn='tanh')
        num_inputs = num_neurons

    model.add_layer(num_inputs=num_inputs, num_neurons=1, act_fn='sigmoid')  # the probability of spiral 1
    model.configure(loss='binary_crossentropy', optimizer='adam')
    return model


# The probability of spiral 1 over a grid that covers the points, for the background of a plot
grid_x, grid_y = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))


def boundary(model):
    return model.predict(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)


# Takes a picture of the network's boundary after some epochs, during a single training: an observer of the model,
# which tells it about every forward pass, update and epoch (see notify in feedforward.py, and monitor.py, the
# dashboard's observer)
class Snapshots:
    def __init__(self, epochs):
        self.epochs = epochs
        self.panels = []

    def on_forward(self, model):
        pass

    def on_update(self, model, gradients, learning_rate):
        pass

    def on_metrics(self, model, values):
        if values["epoch"] in self.epochs:
            _, accuracy = model.evaluate(test_points, test_labels)
            self.panels.append(("2 hidden layers of 16", values["epoch"], accuracy, boundary(model)))
            print(f"2 hidden layers of 16: test accuracy {accuracy * 100:.1f}% after {epochs_text(values['epoch'])}")


def epochs_text(epochs):
    return f"{epochs} epoch{'s' if epochs > 1 else ''}"


def train(model):
    model.train(points, labels, epochs=args.epochs, learning_rate=0.01, batch_size=32)


# Without hidden layers: z = w1 * x + w2 * y + b, so the boundary (z = 0, probability 0.5) is a straight line
linear = build([])
network = build([16, 16])
monitor = start_dashboard(args.dashboard, {"2 hidden layers": network})

train(linear)
_, linear_accuracy = linear.evaluate(test_points, test_labels)
print(f"No hidden layer:      test accuracy {linear_accuracy * 100:.1f}% after {epochs_text(args.epochs)}")

# Two hidden layers of 16 neurons, with pictures of the boundary along the training
snapshots = Snapshots(epochs={1, 10, 50, args.epochs})
network.observers.append(snapshots)
train(network)
panels = [("No hidden layer", args.epochs, linear_accuracy, boundary(linear))] + snapshots.panels


figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))

for axis, (name, epochs, accuracy, probabilities) in zip(axes, panels):
    axis.contourf(grid_x, grid_y, probabilities, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.6)
    axis.contour(grid_x, grid_y, probabilities, levels=[0.5], colors="black", linewidths=1.5)  # the boundary
    axis.scatter(*test_points[test_labels == 0].T, s=4, c="tab:blue")
    axis.scatter(*test_points[test_labels == 1].T, s=4, c="tab:red")
    axis.set_title(f"{name}\n{epochs_text(epochs)}: {accuracy * 100:.1f}% right")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")

figure.suptitle("Probability of the red spiral (background) and the decision boundary at 0.5 (black line)")
plt.tight_layout(rect=(0, 0, 1, 0.9))  # leaves room for the title above the panels
figure_filepath = project_path('spirals.png')
plt.savefig(figure_filepath, dpi=120)
print(f"\nDecision boundaries saved to {display_path(figure_filepath)}")
keep_dashboard_open(monitor)
