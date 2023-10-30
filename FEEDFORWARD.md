# Feedforward

Basic implementation of a feedforward neural network. The network can be customized with various activation functions and multiple layers to suit different tasks.

## Requirements

- `numpy`

## Features

- **Custom Activation Functions**: Supports commonly used activation functions including `ReLU`, `Sigmoid`, `Softmax`, and `Linear`.

- **Modularity**: Each layer and neuron is modularly constructed to allow flexibility in building the network architecture.

- **Verbosity Option**: For those who wish to see the detailed output of each layer and neuron during the forward pass, a verbosity option is included.

## Usage

1. **Initialization**:
   ```python
   ff = FeedForward(verbose=True)
   ```

2. **Add Layers**:
   Add layers, specifying the number of inputs, number of neurons, and the activation function.
   ```python
   ff.add_layer(num_inputs=3, num_neurons=5, act_fn="relu")
   ff.add_layer(num_inputs=5, num_neurons=1, act_fn="sigmoid")
   ```

3. **Forward Pass**:
   Run a forward pass through the network using the `forward` method.
   ```python
   input_values = [0.5, 0.1, 0.3]
   output = ff.forward(input_values)
   print(output)
   ```

