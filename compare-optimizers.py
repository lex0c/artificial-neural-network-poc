import matplotlib.pyplot as plt
import numpy as np

from cli import display_path, project_path, training_args
from data import set_seed
from feedforward import FeedForward
from mnist import load_mnist


# Compares the optimizers of optimizers.py (SGD, momentum, RMSprop and Adam) on the same task: 10,000 MNIST images and
# the same network, starting from the same weights. Each optimizer has its own good learning rate, so every one is
# trained with a range of them. optimizers.png shows:
#   - the validation loss at the end, for each learning rate: how far each optimizer gets, and how wide the range of
#     learning rates that work is
#   - the validation loss after each epoch, with the best learning rate of each optimizer: how fast it gets there

args = training_args("Compares SGD, momentum, RMSprop and Adam on MNIST, over a range of learning rates.",
                     dashboard=False, epochs=8)  # 32 trainings one after the other: nothing to follow live

(train_images, train_labels), _ = load_mnist()
images, labels = train_images[:10000], train_labels[:10000]            # a part of the training images, to be quick
val_images, val_labels = train_images[-5000:], train_labels[-5000:]    # not used for training

# One seed for every run: the same initial weights and order of the samples for every optimizer and learning rate
seed = args.seed if args.seed is not None else int(np.random.randint(2 ** 31))
print(f"Seed {seed} (to repeat this run: --seed {seed})\n")

OPTIMIZERS = ("sgd", "momentum", "rmsprop", "adam")
LEARNING_RATES = (1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)


# The validation loss before the training and after each epoch
def train(optimizer, learning_rate):
    set_seed(seed)
    model = FeedForward()
    model.add_layer(num_inputs=784, num_neurons=64, act_fn='relu')
    model.add_layer(num_inputs=64, num_neurons=10, act_fn='softmax')
    model.configure(loss='sparse_categorical_crossentropy', optimizer=optimizer)

    initial_loss, _ = model.evaluate(val_images, val_labels)
    history = model.train(images, labels, epochs=args.epochs, learning_rate=learning_rate, batch_size=64,
                          validation_data=(val_images, val_labels))
    return [initial_loss] + history["val_loss"]


curves = {}  # (optimizer, learning rate) -> validation losses
print(f"Validation loss after {args.epochs} epochs, for each learning rate:")
print("              " + "".join(f"{rate:>9}" for rate in LEARNING_RATES))

for optimizer in OPTIMIZERS:
    for rate in LEARNING_RATES:
        curves[optimizer, rate] = train(optimizer, rate)
    print(f"{optimizer:>12}: " + "".join(f"{curves[optimizer, rate][-1]:9.3f}" for rate in LEARNING_RATES))

best = {optimizer: min(LEARNING_RATES, key=lambda rate: np.nan_to_num(curves[optimizer, rate][-1], nan=np.inf))
        for optimizer in OPTIMIZERS}
print("\nBest learning rate: " + ", ".join(f"{optimizer} {best[optimizer]}" for optimizer in OPTIMIZERS))


figure, (by_rate, by_epoch) = plt.subplots(1, 2, figsize=(13, 4.8))

for optimizer in OPTIMIZERS:
    final = [curves[optimizer, rate][-1] for rate in LEARNING_RATES]
    line, = by_rate.plot(LEARNING_RATES, final, marker="o", label=optimizer)
    by_rate.plot(best[optimizer], curves[optimizer, best[optimizer]][-1], marker="*", markersize=14, color=line.get_color())
    by_epoch.plot(range(args.epochs + 1), curves[optimizer, best[optimizer]], marker="o",
                  label=f"{optimizer} (learning rate {best[optimizer]})")

by_rate.set_xscale("log")
by_rate.set_ylim(0, 2.5)
by_rate.set_xlabel("learning rate")
by_rate.set_ylabel(f"validation loss after {args.epochs} epochs")
by_rate.set_title("Each optimizer works best with its own learning rate (star)")
by_rate.legend()

by_epoch.set_ylim(0, 0.6)
by_epoch.set_xlabel("epoch")
by_epoch.set_ylabel("validation loss")
by_epoch.set_title("With its best learning rate (every run starts at a loss of 2.3)")
by_epoch.legend()

plt.tight_layout()
figure_filepath = project_path('optimizers.png')
plt.savefig(figure_filepath, dpi=120)
print(f"Comparison saved to {display_path(figure_filepath)}")
