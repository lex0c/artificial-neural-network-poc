import random

from cli import load_trained_model, project_path


# Uses the model saved by train.py to predict y = 2x - 1 for a few random values of x.

model = load_trained_model(project_path('models', 'simple_linear_function.joblib'), 'train.py')
model.summary()

xs = [random.uniform(-10.0, 10.0) for _ in range(3)]

print('X:', xs)
print('Model predict (Y):', model.predict([[x] for x in xs]).ravel())
print('Calculated result (Y):', [2 * x - 1 for x in xs])
