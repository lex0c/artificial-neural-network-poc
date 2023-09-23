from feedforward import neuron, layer, FeedForward
from etc import relu, sigmoid


verbose = True

input_values = [2, -3]
neuron_weights = [0.5, -0.5]
neuron_bias = 1
layer_weights = [[0.5, -0.5], [0.7, 0.1], [0.1, -0.1]]
layer_biases = [1.0, 2.0, -0.1]


print(neuron(input_values, neuron_weights, neuron_bias, relu))
print(layer(input_values, layer_weights, layer_biases, relu, verbose))


network = FeedForward(verbose=True)
network.add_layer(layer_weights, layer_biases, relu)
network.add_layer([[0.5, 0.5, 0.7]], [1], sigmoid)

output = network.forward(input_values)

print("FeedForward Output: ", output)


