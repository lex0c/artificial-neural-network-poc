import numpy as np

from feedforward import FeedForward


# Y = 2X - 1
xs = [[x] for x in np.random.uniform(low=-10.0, high=10.0, size=200)]
ys = [[2*x[0]-1] for x in xs]

test_xs = [[x] for x in np.random.uniform(low=-20.0, high=20.0, size=50)]
test_ys = [[2*x[0]-1] for x in test_xs]

model = FeedForward(verbose=False)
model.add_layer(num_inputs=1, num_neurons=1, act_fn='linear')

model.summary()

model.train(xs, ys, epochs=500, learning_rate=0.0001)

test_loss, test_accuracy = model.evaluate(test_xs, test_ys)

print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.2f}")

model.save('models/simple_linear_function.joblib')

