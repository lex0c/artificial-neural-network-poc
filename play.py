from feedforward import neuron, layer, FeedForward
from etc import save_model, load_model, normalize_minmax, one_hot_encode


verbose = True
epsilon = 0.1

classes = ['A', 'B', 'C']
print(classes, one_hot_encode(classes, 'B'))

input_values = [2, -3]
print(input_values, normalize_minmax(input_values))


neuron_weights = [0.5, -0.5]
neuron_bias = 1
layer_weights = [[0.5, -0.5], [0.7, 0.1], [0.1, -0.1]]
layer_biases = [1.0, 2.0, -0.1]


#print(neuron(input_values, neuron_weights, neuron_bias, relu))
#print(layer(input_values, layer_weights, layer_biases, relu, verbose))


network = FeedForward(layers=load_model('play'), verbose=True)
#network.add_layer(num_inputs=len(input_values), num_neurons=4, act_fn="relu")
#network.add_layer(num_inputs=4, num_neurons=4, act_fn="relu")
#network.add_layer(num_inputs=4, num_neurons=1, act_fn="sigmoid")

print(network.layers)

#save_model('play', network.layers)

output = network.forward(normalize_minmax(input_values))

print("FeedForward Output: ", output)


