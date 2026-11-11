# FeedForward

`FeedForward` (in `feedforward.py`) is a feedforward neural network written with NumPy only. This page explains how it
computes and how it learns, works out one training step by hand, and then documents how to use it.

## How the network computes: the forward pass

A network is a stack of **dense layers**. Each neuron `j` of a layer computes a weighted sum of all its inputs plus a
bias, and passes it through an **activation function** `f`:

```
z[j]      = weights[j, 0] * inputs[0] + weights[j, 1] * inputs[1] + ... + biases[j]
output[j] = f(z[j])
```

The outputs of a layer are the inputs of the next one, and the outputs of the last layer are the predictions. In
code, a layer is one matrix product (`layer_forward` in `feedforward.py`), and `forward_layers` runs the layers in
order, keeping the inputs, `z` and output of each one in `model.cache`, because the backward pass needs them.

Without activation functions, a stack of layers would still compute a linear function of its input. The
non-linearity of `f` is what lets a network learn curved relationships.

## How the network learns: gradient descent

1. **Loss.** A loss function measures how wrong the predictions are, as one number. Training means finding the weights
   that make it small.
2. **Gradient.** For each weight, the derivative of the loss with respect to it says how the loss changes when that
   weight changes a little. All these derivatives together are the gradient.
3. **Update.** The gradient points to where the loss grows the fastest, so each weight takes a small step the other
   way: `weight = weight - learning_rate * gradient`. The learning rate sets the size of the step. Too small learns
   slowly, and too large overshoots and can diverge. This is plain gradient descent (SGD); the
   [Adam optimizer](#optimizers-optimizerspy) adapts the step of each weight.

Repeating these steps over the data lowers the loss. This is `train_step` in `feedforward.py`:

```python
predictions = self.forward_layers(inputs)                      # 1. forward pass
loss = loss_fn(self.configs["loss"], targets, predictions)     # 2. loss
output_delta = self.output_delta(targets, predictions)         # 3. gradient at the output layer
gradients = self.backward(output_delta, l1_lambda, l2_lambda)  # 4. backpropagation
self.update(gradients, learning_rate)                          # 5. gradient descent
```

### Backpropagation: the chain rule

The loss depends on a weight only through the layers after it, so its derivative is a product of simpler
derivatives (the **chain rule**). For a weight `w` of a neuron with sum `z` and output `y = f(z)`:

```
dL/dw = dL/dy * dy/dz * dz/dw
      = dL/dy * f'(z) * input
```

The part `dL/dz = dL/dy * f'(z)` is called **delta**. Backpropagation computes it for the output layer first
(`output_delta`) and then walks back one layer at a time (`backward`). The gradient for the inputs of a layer,
`dL/dinputs[k] = sum over j of delta[j] * weights[j, k]`, is the `dL/dy` of the previous layer, so it goes through that
layer's activation (`activation_backward` in `activations.py`) to become its delta, and so on down to the first layer.

`backward` computes every gradient before `update` changes any weight, so each gradient uses the same weights that
produced the predictions.

## One training step, worked out by hand

This is the model of `train.py`: one neuron with a linear activation, `y = w * x + b`, that should learn `y = 2x - 1`.
It starts with `w = 0.5` and `b = 0`, trains on the sample `x = 3` (target `2 * 3 - 1 = 5`), and uses a learning rate of
`0.01` and the mean squared error.

| Step | Math | Value | Code |
| --- | --- | --- | --- |
| 1. Forward | `z = w * x + b` | `0.5 * 3 + 0 = 1.5` | `layer_forward` |
| | `y_hat = f(z) = z` (linear) | `1.5` | `linear` |
| 2. Loss | `L = (y_hat - y)^2` | `(1.5 - 5)^2 = 12.25` | `mse_loss` |
| 3. Output delta | `dL/dy_hat = 2 * (y_hat - y)` | `2 * (1.5 - 5) = -7` | `mse_loss_derivative` |
| | `delta = dL/dy_hat * f'(z)`, with `f'(z) = 1` | `-7` | `output_delta` |
| 4. Backward | `dL/dw = delta * x` | `-7 * 3 = -21` | `backward` |
| | `dL/db = delta` | `-7` | `backward` |
| 5. Update | `w = w - 0.01 * dL/dw` | `0.5 + 0.21 = 0.71` | `update` |
| | `b = b - 0.01 * dL/db` | `0 + 0.07 = 0.07` | `update` |

With the new weights the prediction is `0.71 * 3 + 0.07 = 2.2` and the loss `(2.2 - 5)^2 = 7.84`, down from `12.25`.
Both gradients were negative, which means a larger `w` and `b` lower the loss, so the update increased them. After
enough steps on many samples, `train.py` finds `w = 2` and `b = -1`.

The test `test_worked_example_in_feedforward_md` in `test_feedforward.py` runs this exact step, so these numbers stay
in sync with the code.

## Batches and matrix shapes

Training on one sample at a time gives noisy steps, and computing the gradient over the whole dataset is slow. A
**mini-batch** is the middle ground: `train` splits the data into batches of `batch_size` samples, and each batch makes
one update with the gradient of the mean loss over its samples. One pass over all the batches is an **epoch**.

Every sample of a batch goes through the layers at once, as one row of a matrix. For the weather model
(`train-weather.py`: 3 inputs, a hidden layer of 16 ReLU neurons, 4 softmax outputs) and a batch of 32 samples:

```
Forward                                       Backward
X        (32, 3)    the batch                 delta2 = (P - Y) / 32                (32, 4)
W1       (16, 3)    one row per neuron        dW2 = delta2^T . A1                  (4, 16)  same shape as W2
Z1 = X . W1^T + b1  (32, 16)                  db2 = sum of the rows of delta2      (4,)
A1 = relu(Z1)       (32, 16)                  dA1 = delta2 . W2                    (32, 16)
W2       (4, 16)                              delta1 = dA1 * relu'(Z1)             (32, 16)
Z2 = A1 . W2^T + b2 (32, 4)                   dW1 = delta1^T . X                   (16, 3)  same shape as W1
P = softmax(Z2)     (32, 4)   probabilities   db1 = sum of the rows of delta1      (16,)
```

The product `delta^T . inputs` sums the gradient of each sample, and the delta already includes the division by the
batch size, so the result is the gradient of the mean loss.

**Shuffling.** With `shuffle=True` (the default), every epoch goes through the samples in a new random order. The
weather data is generated grouped by condition: without shuffling, each batch would hold mostly one class and push the
model towards it.

## Building blocks

### Activation functions (`activations.py`)

| Name | Formula | Derivative | Typical use |
| --- | --- | --- | --- |
| `relu` | `max(0, z)` | `1` if `z > 0`, else `0` | hidden layers |
| `sigmoid` | `1 / (1 + e^-z)`, in `(0, 1)` | `s * (1 - s)` | a probability, e.g. binary classification |
| `tanh` | `(e^z - e^-z) / (e^z + e^-z)`, in `(-1, 1)` | `1 - tanh(z)^2` | hidden layers (the usual choice before ReLU) |
| `linear` | `z` | `1` | output of a regression, Q-values |
| `softmax` | `e^z[j] / sum of e^z[k]`, sums to 1 | a Jacobian, see `activation_backward` | output of a multi-class classifier |

### Loss functions (`losses.py`)

| Name | Formula (per sample) | Use |
| --- | --- | --- |
| `mse` | mean of `(y_pred - y_true)^2` | regression |
| `categorical_crossentropy` | `-sum(y_true * log(y_pred))`, with one-hot `y_true` | classification with a softmax output |
| `sparse_categorical_crossentropy` | the same, with class indices as labels | classification with a softmax output |

Cross-entropy is minus the log of the probability given to the true class: close to 0 when that probability is close
to 1, and very large when a confident prediction is wrong. The probabilities are clipped to `[1e-7, 1 - 1e-7]` so that
`log(0)` can't happen; the derivatives ignore this clipping, so the gradient doesn't vanish when a prediction is
completely wrong. With a softmax output, its gradient with respect to `z` simplifies to
`(y_pred - y_true) / batch_size` (`softmax_crossentropy_delta`), which `output_delta` uses directly.

Sigmoid and tanh saturate: for a large `|z|` their output is almost flat, so their derivative is close to 0 and the
gradient fades as it goes back through many such layers. ReLU doesn't saturate for positive `z`, which is why it
replaced them in hidden layers.

### Optimizers (`optimizers.py`)

The optimizer is the last step of `train_step`: it decides how the gradients change the parameters.

- **SGD** (the default): `parameter = parameter - learning_rate * gradient`. The step is proportional to the gradient,
  so parameters with small gradients learn slowly, and one learning rate has to suit all of them.
- **Adam**: keeps, for each parameter, a running mean of its gradients (`m`, a momentum: steps keep their direction when
  the gradients agree) and of its squared gradients (`v`, how large its gradients usually are), and moves it by
  `learning_rate * m / sqrt(v)`. Dividing by `sqrt(v)` gives every parameter a step of about `learning_rate`, whatever
  the scale of its gradients.

```
m = beta1 * m + (1 - beta1) * gradient          beta1 = 0.9
v = beta2 * v + (1 - beta2) * gradient^2        beta2 = 0.999
m_corrected = m / (1 - beta1^t)                 t: number of updates so far
v_corrected = v / (1 - beta2^t)
parameter = parameter - learning_rate * m_corrected / (sqrt(v_corrected) + 1e-8)
```

`m` and `v` start at 0, so they are too small in the first updates; dividing by `1 - beta^t` corrects that. In the
first update the step is `learning_rate * gradient / (|gradient| + 1e-8)`: about `learning_rate` against the
direction of the gradient (unless the gradient is close to `1e-8`). The averages carry over between calls to `train`, which matters when a model is trained a little at a
time (as the DQN agent does).

On MNIST (`train-mnist.py`, 3 seeds, 10 epochs, validation accuracy):

| Optimizer | After 1 epoch | After 3 epochs | After 10 epochs | Test |
| --- | --- | --- | --- | --- |
| Adam, `learning_rate=0.001` | 96.2% | 97.4% | 98.0% | 97.8% |
| SGD, `learning_rate=0.1` | 93.6% | 96.5% | 97.8% | 97.4% |
| SGD, `learning_rate=0.001` | 58.6% | 81.2% | 89.8% | 87.3% |

Adam learns more per epoch and doesn't need a learning rate tuned to the scale of the gradients: with Adam's
`0.001`, SGD learns much more slowly (89.8% after 10 epochs). Each Adam update does more work, though, so an epoch took about twice as long.

### Weight initialization

The weights start random: if they were all equal, every neuron of a layer would compute the same thing and get the
same update forever. Their scale aims to keep the variance of `z` similar from layer to layer, so the signal neither
explodes nor fades: **He** initialization (variance `2 / num_inputs`) for ReLU layers, and **LeCun** initialization
(variance `1 / num_inputs`) for the others. The biases start at 0.

The factor 2 of He makes up for a ReLU zeroing about half of its outputs, so it keeps the variance exactly from one ReLU
layer to the next. Choosing by the layer's own activation, as `add_layer` does, is the usual rule of thumb: the first
layer, whose inputs don't come from a ReLU, doubles the variance, and a non-ReLU layer after a ReLU layer halves it.

### Gradient clipping

Before backpropagation, each value of `dL/dy_pred` is limited to `[-10, 10]` (`clip_gradients`), so a single huge
error can't produce an update large enough to make the training diverge. This gradient already includes the division
by the batch size (and, for `mse`, by the number of outputs), so with large batches an error has to be much larger to
be clipped: with `mse`, one output and `batch_size=1`, it happens when `|y_pred - y_true| > 5`. It doesn't apply to
softmax with cross-entropy, whose gradient is already between `-1 / batch_size` and `1 / batch_size`.

## Usage

### Create a model

```python
from feedforward import FeedForward

model = FeedForward()
model.add_layer(num_inputs=1, num_neurons=1, act_fn='linear')
model.configure(loss='mse')  # optimizer='sgd' by default; optimizer='adam' for Adam
model.summary()
```

This is the model of `train.py`, which the next examples train. `num_inputs` of each layer must match `num_neurons`
of the previous one (or the number of features, for the first layer). A classifier with 3 features and 4 classes, as
in `train-weather.py`:

```python
classifier = FeedForward()
classifier.add_layer(num_inputs=3, num_neurons=16, act_fn='relu')
classifier.add_layer(num_inputs=16, num_neurons=4, act_fn='softmax')
classifier.configure(loss='categorical_crossentropy')
```

### Normalize the inputs

Features with very different scales (e.g. a temperature around 20 and a humidity around 80) get very different
gradients, which slows down or stalls the training. Before training, fit a min-max normalization to the training
inputs:

```python
from weather import generate_weather_dataset

features, labels = generate_weather_dataset(100)
classifier.fit_normalization(features)
```

It maps each feature of the training data to `[0, 1]` and is stored in the model configs, so `train`, `predict` and
`evaluate` scale every input with the same values, also after the model is saved and loaded. Always pass the raw
inputs. Inputs in `[0, 1]` give smaller gradients, so a larger learning rate is usually needed.

### Train

```python
inputs = [[-1.70], [-2.34], [-4.75], [-5.56], [9.08]]
targets = [[-4.40], [-5.68], [-10.50], [-12.12], [17.16]]  # y = 2x - 1

history = model.train(inputs, targets, epochs=500, learning_rate=0.01, batch_size=1, verbose=True)
```

- **inputs**: one row per sample, one column per feature. A flat list is accepted for a model with one input.
- **targets**: one row per sample, one column per output (one-hot for `categorical_crossentropy`, class indices for
  `sparse_categorical_crossentropy`). A flat list is accepted for a model with one output.
- **epochs**: number of passes over the whole dataset.
- **learning_rate**: size of each gradient descent step.
- **batch_size**: number of samples per update (default 32). Use 1 to update after every sample.
- **l1_lambda**, **l2_lambda**: regularization (default 0). They add the gradient of the penalty
  `l1 * sum(|w|) + l2 / 2 * sum(w^2)` on the weights (not the biases), which pulls them towards 0 and reduces
  overfitting. The loss reported by `train` and `evaluate` doesn't include the penalty.
- **shuffle**: a new random order of the samples every epoch (default True).
- **verbose**: print the progress and the loss of each epoch (default False).
- **validation_data**: optional `(inputs, targets)` not used for training. Its loss (and accuracy, for classification)
  is calculated after each epoch, to see whether the model generalizes or starts to overfit.

`train` returns `model.history`, with the loss of each epoch (`loss`) and the validation metrics (`val_loss`,
`val_accuracy`).

### Evaluate

```python
test_inputs = [[0.0], [5.0], [-8.0]]
test_targets = [[-1.0], [9.0], [-17.0]]

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print(test_loss, test_accuracy)  # a small loss, and None: there is no accuracy for regression
```

The accuracy depends on the output layer:

- **softmax**, several **sigmoid** outputs or class indices (`sparse_categorical_crossentropy`): the output with the
  highest value is the predicted class.
- a single **sigmoid** output (binary classification): above 0.5 is class 1.
- other outputs, such as **linear** for regression: there is no accuracy, so it is `None`.

For regression, use the error metrics instead:

```python
metrics = model.regression_metrics(test_inputs, test_targets)
print(metrics["mae"], metrics["rmse"], metrics["r2"])
```

- **mae**: mean absolute error, in the units of the targets.
- **rmse**: root mean squared error. It penalizes large errors more than `mae`.
- **r2**: coefficient of determination. 1 is perfect, and 0 is as good as always predicting the mean of the targets.

### Predict

```python
predictions = model.predict([[6.18]])
print(predictions)  # about [[11.36]] once trained on y = 2x - 1
```

### Save and load

```python
model.save('models/my_model.joblib')

from feedforward import load_model

model = load_model('models/my_model.joblib')
```

The file keeps the weights, biases and activation of each layer, the loss function and the input normalization.

### Clone

```python
from feedforward import clone_model

new_model = clone_model(model)          # an independent copy
new_model.set_layers(model.get_layers())  # copy the weights of model into new_model again later
```

The DQN agent uses this for its target network (see [DQN.md](DQN.md)).

## Checking the gradients numerically

`test_feedforward.py` checks backpropagation without trusting its formulas: `numerical_gradients` nudges every weight
by a tiny amount up and down and measures how the loss changes, `(L(w + eps) - L(w - eps)) / (2 * eps)`. If the
gradients from `backward` match these numbers, backpropagation is right. It is a useful exercise to break a formula in
`backward` and watch the test fail.
