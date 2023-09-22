from nn import neuron
from etc import relu


input_values = [2, -3]
weights = [0.5, -0.5]
bias = 1


print(neuron(input_values, weights, bias, relu))


