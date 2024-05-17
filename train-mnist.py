import numpy as np
import tensorflow as tf

from feedforward import FeedForward


mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

n_train_samples = 200
n_test_samples = 50

training_images = training_images[:n_train_samples]
training_labels = training_labels[:n_train_samples]

# Normalizing values between 0 and 1
training_images = training_images / 255.0
test_images = test_images / 255.0

# Flatten images of 2D to 1D
flattened_training_images = training_images.reshape((training_images.shape[0], -1))
flattened_test_images = test_images.reshape((test_images.shape[0], -1))

# one-hot encoding
training_labels = tf.keras.utils.to_categorical(training_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

model = FeedForward()
model.add_layer(num_inputs=flattened_training_images.shape[1], num_neurons=256, act_fn='relu')
model.add_layer(num_inputs=256, num_neurons=256, act_fn='relu')
model.add_layer(num_inputs=256, num_neurons=training_labels.shape[1], act_fn='softmax')
model.summary()

model.train(flattened_training_images, training_labels, epochs=5, learning_rate=0.001, batch_size=32, use_granular_update=True, verbose=True)

test_loss, test_accuracy = model.evaluate(flattened_test_images, test_labels)

print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.2f}")

model.save('models/mnist_classification.joblib')

