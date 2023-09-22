def neuron(values, weights, bias, act_fn):
    assert len(values) == len(weights), "Sizes of values and weights must be the same!"

    # Linear combination of inputs and weights, adding bias
    linear_combination = sum([i*w for i, w in zip(values, weights)]) + bias

    # Pass the linear combination through the activation function
    output = act_fn(linear_combination)

    return output


