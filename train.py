import numpy as np

from data import set_seed
from feedforward import FeedForward


# Linear regression with a single neuron: learns y = 2x - 1 from examples. The neuron computes y = w * x + b, so the
# training should find w = 2 and b = -1. FEEDFORWARD.md works out one training step of this model by hand.

SEED = None  # set an integer (e.g. 0) to make the run reproducible
set_seed(SEED)

xs = [[x] for x in np.random.uniform(low=-10.0, high=10.0, size=200)]
ys = [[2*x[0]-1] for x in xs]

# The test inputs go beyond the training range: a linear model extrapolates without problems
test_xs = [[x] for x in np.random.uniform(low=-20.0, high=20.0, size=50)]
test_ys = [[2*x[0]-1] for x in test_xs]

model = FeedForward()
model.add_layer(num_inputs=1, num_neurons=1, act_fn='linear')
model.configure(loss="mse")
model.summary()

model.train(xs, ys, epochs=500, learning_rate=0.0001, batch_size=1, verbose=True)  # batch_size=1: one update per sample

test_loss, _ = model.evaluate(test_xs, test_ys)  # no accuracy for regression
metrics = model.regression_metrics(test_xs, test_ys)

print(f"Learned: w = {model.layers[0]['weights'][0, 0]:.4f}, b = {model.layers[0]['biases'][0]:.4f} (expected 2 and -1)")
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {metrics['mae']:.4f}")
print(f"Test RMSE: {metrics['rmse']:.4f}")
print(f"Test R2: {metrics['r2']:.4f}")

model.save('models/simple_linear_function.joblib')
