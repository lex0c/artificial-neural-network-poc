# Neural Network POC

A small neural network framework written with NumPy only, to learn how neural networks and Deep Q-Learning work.
Every step, from the forward pass to backpropagation and the DQN training loop, is plain code meant to be read.

## Setup

Requires Python 3.12 or newer.

```sh
pip install -r requirements.txt
python -m unittest -v   # optional: checks that everything works
```

## Reading order

1. **[FEEDFORWARD.md](FEEDFORWARD.md)**: how a network computes and learns (forward pass, loss, backpropagation,
   gradient descent), with one training step worked out by hand.
2. **The framework**:
   - `feedforward.py`: the `FeedForward` model. Start with `train_step`, which calls the five steps of a training step.
   - `activations.py`: ReLU, sigmoid, linear and softmax, and their derivatives.
   - `losses.py`: mean squared error and cross-entropy, and their derivatives.
   - `data.py`: one-hot encoding, normalization, batches and seeds.
3. **The examples**, from the simplest (below).
4. **[DQN.md](DQN.md)** and `dqn.py`: reinforcement learning with a Deep Q-Network.
5. **The tests**: `test_feedforward.py` checks backpropagation against gradients measured numerically, and
   `test_dqn.py` computes DQN and Double DQN targets on a tiny example you can follow by hand.

The `models/` folder has trained models, so the `run` scripts work right away.

## Examples

### 1. Linear regression: y = 2x - 1

A single neuron learns `y = w * x + b`, and training should find `w = 2` and `b = -1`. FEEDFORWARD.md works out one
of its training steps by hand.

```sh
python train.py   # trains and saves models/simple_linear_function.joblib
python run.py     # predicts a few values with it
```

### 2. Classification: weather conditions

A network with a hidden layer predicts the weather condition (Sunny, Cloudy, Rainy or Snowy) from the temperature,
humidity and wind speed. It shows softmax, cross-entropy, input normalization and validation. The synthetic data
comes from `weather.py`.

```sh
python train-weather.py   # trains and saves models/weather.joblib
python run-weather.py     # per-class report and confusion-matrix.png, on new data
```

### 3. Reinforcement learning: CartPole

A [Deep Q-Network](DQN.md) learns to balance a pole on a cart
([CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole)).

```sh
python train-cartpole.py          # trains and saves the best model to models/cartpole.joblib
python run-cartpole.py            # watch it play (models/cartpole.joblib, or another file as argument)
python show-cartpole-metrics.py   # plots metrics/cartpole.csv to cartpole_<run>.png
```

`finetune-cartpole.py` continues training `models/cartpole.joblib` with less exploration, and saves the best model to
`models/cartpole-checkpoint.joblib` (watch it with `python run-cartpole.py models/cartpole-checkpoint.joblib`) and
the last one to `models/cartpole-finetuned.joblib`.

The training scripts have two settings at the top: `SEED`, to make a run reproducible, and, for CartPole, `RENDER`,
to watch the training (much slower).
