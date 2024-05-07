import random

from feedforward import load_model


model = load_model('models/simple_linear_function.joblib')
model.summary()

x1 = random.uniform(-10.0, 10.0)
x2 = random.uniform(-10.0, 10.0)
x3 = random.uniform(-10.0, 10.0)

expected_result1 = 2*x1-1
expected_result2 = 2*x2-1
expected_result3 = 2*x3-1

print('X:', [x1, x2, x3])
print('Model predict (Y):', model.predict([[x1], [x2], [x3]]))
print('Calculated result (Y):', [expected_result1, expected_result2, expected_result3])

