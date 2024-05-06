import random

from feedforward import load_model


model = load_model('models/simple_linear_function.json')
model.summary()

x = random.uniform(-10.0, 10.0)
expected_result = 2*x-1

print('X:', x)
print('Model predict (Y):', model.predict([x]))
print('Calculated result (Y):', expected_result)

