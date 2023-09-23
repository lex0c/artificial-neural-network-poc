def neuron(values, weights, bias, act_fn):
    if len(values) != len(weights):
        raise ValueError("Sizes of values and weights must be the same!")

    linear_combination = 0

    # Linear combination of values and weights
    for value, weight in zip(values, weights):
        linear_combination += value * weight

    # Add bias
    linear_combination += bias

    # Pass the linear combination through the activation function
    output = act_fn(linear_combination)

    return output


def layer(values, weights, biases, act_fn, verbose=False):
    if len(weights) != len(biases):
        raise ValueError("Sizes of weights and biases must be the same!")

    # Number of neurons
    size = len(weights)

    # Init empty array
    output = [0]*size

    # Run neurons
    for i in range(size):
        output[i] = neuron(values, weights[i], biases[i], act_fn)

        if verbose:
            print(f'Neuron {i+1}: {output[i]}')

    return output


