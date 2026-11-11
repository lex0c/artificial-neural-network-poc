import numpy as np


# Synthetic weather data for the classification example: each condition has its own range of temperature, humidity
# and wind speed. Some ranges touch at their edges, so a few samples near the boundaries are hard to tell apart.
WEATHER_CONDITIONS = {
    'Sunny': {'temp': (20, 35), 'humidity': (10, 50), 'wind': (0, 20)},
    'Cloudy': {'temp': (10, 20), 'humidity': (50, 90), 'wind': (0, 10)},
    'Rainy': {'temp': (5, 15), 'humidity': (80, 100), 'wind': (10, 25)},
    'Snowy': {'temp': (-5, 0), 'humidity': (50, 100), 'wind': (5, 15)}
}

CLASS_NAMES = list(WEATHER_CONDITIONS)


# Generates num_samples per condition: the features [temperature, humidity, wind speed], drawn uniformly from the
# ranges of the condition, and the index of the condition as the label. The samples come grouped by condition.
def generate_weather_dataset(num_samples=10):
    features = []
    labels = []

    for label, ranges in enumerate(WEATHER_CONDITIONS.values()):
        for _ in range(num_samples):
            temp = np.random.uniform(*ranges['temp'])
            humidity = np.random.uniform(*ranges['humidity'])
            wind = np.random.uniform(*ranges['wind'])

            features.append([temp, humidity, wind])
            labels.append(label)

    return np.array(features), np.array(labels)
