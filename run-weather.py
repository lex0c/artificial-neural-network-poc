import random
import numpy as np

from feedforward import load_model


model = load_model('models/weather.joblib')
model.summary()


new_weather_data = [10, 85, 15]  # Example: [Temperature, Humidity, Wind Speed]

predicted_condition = model.predict(new_weather_data)

print("Predicted Weather Condition:", predicted_condition, np.argmax(predicted_condition[0]))

