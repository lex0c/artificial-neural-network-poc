import random

from cli import keep_dashboard_open, load_trained_model, project_path, run_args, start_dashboard


# Uses the model saved by train.py to predict y = 2x - 1 for a few random values of x.

args = run_args("Predicts y = 2x - 1 with the model saved by train.py.")
model = load_trained_model(project_path('models', 'simple_linear_function.joblib'), 'train.py')
model.summary()
monitor = start_dashboard(args.dashboard, {"linear regression": model})

xs = [random.uniform(-10.0, 10.0) for _ in range(3)]

print('X:', xs)
print('Model predict (Y):', model.predict([[x] for x in xs]).ravel())
print('Calculated result (Y):', [2 * x - 1 for x in xs])
keep_dashboard_open(monitor)
