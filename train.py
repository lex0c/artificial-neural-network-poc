import numpy as np

from feedforward import FeedForwardAutoencoder
from etc import mse_loss


# Generate synthetic normal data
def generate_normal_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        sample = np.random.normal(loc=0.5, scale=0.1, size=10)  # 10 features
        data.append(sample)
    return np.array(data)

# Generate synthetic anomalous data
def generate_anomalous_data(num_samples=100):
    data = []
    for _ in range(num_samples):
        sample = np.random.uniform(low=0, high=1, size=10)  # 10 features, different distribution
        data.append(sample)
    return np.array(data)

# Training the autoencoder
def train_autoencoder(autoencoder, normal_data, epochs=100):
    for epoch in range(epochs):
        total_loss = 0

        for sample in normal_data:
            total_loss += autoencoder.train(sample)

        avg_loss = total_loss / len(normal_data)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# Detecting anomalies
def detect_anomalies(autoencoder, data, threshold):
    anomalies = []

    for sample in data:
        reconstructed_sample = autoencoder.forward(sample)
        loss = mse_loss(sample, reconstructed_sample)

        if loss > threshold:
            anomalies.append(sample)

    return anomalies

# Usage
normal_data = generate_normal_data()
anomalous_data = generate_anomalous_data()

autoencoder = FeedForwardAutoencoder(learning_rate=0.01)
autoencoder.add_layer(num_inputs=10, num_neurons=5, act_fn="sigmoid")
autoencoder.add_layer(num_inputs=5, num_neurons=10, act_fn="sigmoid")

# Train autoencoder with normal data
train_autoencoder(autoencoder, normal_data)

# Detect anomalies in a mixed dataset
mixed_data = np.concatenate((normal_data, anomalous_data), axis=0)
anomalies = detect_anomalies(autoencoder, mixed_data, threshold=0.1)
print(f"Number of anomalies detected: {len(anomalies)}")

