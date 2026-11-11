from data import set_seed
from feedforward import FeedForward
from mnist import load_mnist


# Classification of handwritten digits (MNIST): a network with one hidden layer learns to recognize the digit in a
# 28 x 28 image, and the model is saved to models/mnist.joblib. The first run downloads the dataset (about 11 MB).

SEED = None  # set an integer (e.g. 0) to make the run reproducible
set_seed(SEED)

(train_images, train_labels), (test_images, test_labels) = load_mnist()

# Validation: the last 5,000 training images, checked after each epoch and not used for training
train_images, val_images = train_images[:-5000], train_images[-5000:]
train_labels, val_labels = train_labels[:-5000], train_labels[-5000:]


model = FeedForward()
model.add_layer(num_inputs=784, num_neurons=128, act_fn='relu')    # one input per pixel (already scaled to [0, 1])
model.add_layer(num_inputs=128, num_neurons=10, act_fn='softmax')  # one probability per digit
model.configure(loss='sparse_categorical_crossentropy', optimizer='adam')  # the labels are digits, not one-hot vectors
model.summary()

model.train(train_images, train_labels, epochs=10, learning_rate=0.001, batch_size=64, verbose=True, validation_data=(val_images, val_labels))

test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

model.save('models/mnist.joblib')
