import numpy as np

from feedforward import FeedForward
from etc import to_categorical


def generate_weather_dataset(num_samples=10):
    weather_data = {
        'Sunny': {'temp': (20, 35), 'humidity': (10, 50), 'wind': (0, 20)},
        'Cloudy': {'temp': (10, 20), 'humidity': (50, 90), 'wind': (0, 10)},
        'Rainy': {'temp': (5, 15), 'humidity': (80, 100), 'wind': (10, 25)},
        'Snowy': {'temp': (-5, 0), 'humidity': (50, 100), 'wind': (5, 15)}
    }

    features = []
    labels = []

    # Define label indices
    label_indices = {label: idx for idx, label in enumerate(weather_data.keys())}

    # Generate samples
    for label, ranges in weather_data.items():
        for _ in range(num_samples):
            temp = np.random.uniform(*ranges['temp'])
            humidity = np.random.uniform(*ranges['humidity'])
            wind = np.random.uniform(*ranges['wind'])

            features.append([temp, humidity, wind])
            labels.append(label_indices[label])

    # Convert to numpy arrays
    features = np.array(features)

    return features, labels, weather_data.keys()


features, labels, labels_raw = generate_weather_dataset(100)
labels_normalized = to_categorical(labels, num_classes=len(labels_raw))

test_features, test_labels, test_labels_raw = generate_weather_dataset(20)
test_labels_normalized = to_categorical(test_labels, num_classes=len(test_labels_raw))


model = FeedForward()
model.add_layer(num_inputs=features.shape[1], num_neurons=16, act_fn='relu')
model.add_layer(num_inputs=16, num_neurons=len(labels_raw), act_fn='softmax')
model.configure(loss='categorical_crossentropy')
model.summary()

model.train(features, labels_normalized, epochs=100, learning_rate=0.001, use_granular_update=True, verbose=True)

test_loss, test_accuracy = model.evaluate(test_features, test_labels_normalized)

print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.2f}")

model.save('models/weather.joblib')

