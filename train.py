import numpy as np

from feedforward import FeedForward


# Y = 2X - 1
xs = [[x] for x in np.random.uniform(low=-10.0, high=10.0, size=100)]
ys = [[2*x[0]-1] for x in xs]


model = FeedForward(verbose=False)
model.add_layer(num_inputs=1, num_neurons=1, act_fn='linear')

model.summary()

model.train(xs, ys, epochs=500, learning_rate=0.0001)

model.save('models/simple_linear_function.json')

