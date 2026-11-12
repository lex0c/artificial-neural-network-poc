import matplotlib.pyplot as plt
import numpy as np

from cli import display_path, keep_dashboard_open, project_path, start_dashboard, training_args
from data import set_seed
from feedforward import FeedForward
from mnist import load_mnist


# An autoencoder: a network that learns to reproduce its input, through a narrow layer in the middle. Here the 784
# pixels of an MNIST digit go down to only 2 numbers (the code) and back up to 784. To rebuild the image from 2
# numbers, the network has to find what matters most in a digit, and it does so without ever seeing the labels: they
# are only used to color the plot. autoencoder.png shows:
#   - test digits and their reconstructions from 2 numbers
#   - the code of each test digit as a point: the digits gather in regions of their own
#   - what the second half of the network (the decoder) draws for each point of a grid over the codes

args = training_args("An autoencoder that squeezes MNIST digits into 2 numbers and rebuilds them.", epochs=10)
set_seed(args.seed)

(train_images, _), (test_images, test_labels) = load_mnist()

model = FeedForward()
model.add_layer(num_inputs=784, num_neurons=128, act_fn='relu')     # encoder
model.add_layer(num_inputs=128, num_neurons=2, act_fn='linear')     # the code: 2 numbers
model.add_layer(num_inputs=2, num_neurons=128, act_fn='relu')       # decoder
model.add_layer(num_inputs=128, num_neurons=784, act_fn='sigmoid')  # one value between 0 and 1 per pixel
# Each pixel is a value between 0 and 1, compared with the original with binary cross-entropy
model.configure(loss='binary_crossentropy', optimizer='adam')
model.summary()
monitor = start_dashboard(args.dashboard, {"autoencoder": model})

# The targets are the inputs themselves
model.train(train_images, train_images, epochs=args.epochs, learning_rate=0.001, batch_size=64, verbose=True)
test_loss, _ = model.evaluate(test_images, test_images)
print(f"\nReconstruction loss on the test images: {test_loss:.4f}")

# The two halves, as models of their own that share the layers of the trained one
encoder = FeedForward(layers=model.layers[:2])
decoder = FeedForward(layers=model.layers[2:])
codes = encoder.predict(test_images)


figure = plt.figure(figsize=(13, 9.5))
grid = figure.add_gridspec(3, 2, height_ratios=(1, 1, 5))

# 1. The first test image of each digit, and its reconstruction
examples = [np.flatnonzero(test_labels == digit)[0] for digit in range(10)]
for row, (images, name) in enumerate(((test_images[examples], "original"), (model.predict(test_images[examples]), "rebuilt from 2 numbers"))):
    axis = figure.add_subplot(grid[row, :])
    axis.imshow(np.hstack([image.reshape(28, 28) for image in images]), cmap="gray_r")
    axis.set_title(name, fontsize=10, loc="left")
    axis.axis("off")

# 2. The code of each test image, colored by its digit (the network never saw the labels)
axis = figure.add_subplot(grid[2, 0])
points = axis.scatter(codes[:, 0], codes[:, 1], c=test_labels, cmap="tab10", s=2, alpha=0.6)
axis.legend(*points.legend_elements(), title="digit", fontsize=8, markerscale=1.5, loc="best")
axis.set_title("The 2 numbers of each test digit")
low, high = np.percentile(codes, [2, 98], axis=0)
axis.set_xlim(low[0], high[0])
axis.set_ylim(low[1], high[1])

# 3. What the decoder draws over a 15 x 15 grid of codes, in the same region of the plane
steps = 15
xs, ys = np.linspace(low[0], high[0], steps), np.linspace(high[1], low[1], steps)  # y from the top down
drawn = decoder.predict(np.array([[x, y] for y in ys for x in xs])).reshape(steps, steps, 28, 28)
axis = figure.add_subplot(grid[2, 1])
axis.imshow(drawn.transpose(0, 2, 1, 3).reshape(steps * 28, steps * 28), cmap="gray_r",
            extent=(low[0], high[0], low[1], high[1]), aspect="auto")
axis.set_title("What the decoder draws for each point")

plt.tight_layout()
figure_filepath = project_path('autoencoder.png')
plt.savefig(figure_filepath, dpi=110)
print(f"Reconstructions and codes saved to {display_path(figure_filepath)}")
keep_dashboard_open(monitor)
