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
   - `activations.py`: ReLU, sigmoid, tanh, linear and softmax, and their derivatives.
   - `losses.py`: mean squared error and cross-entropy, and their derivatives.
   - `optimizers.py`: plain gradient descent (SGD) and Adam.
   - `data.py`: one-hot encoding, normalization, batches and seeds.
3. **The examples**, from the simplest (below).
4. **Reinforcement learning**:
   - [DQN.md](DQN.md) and `dqn.py`: a Deep Q-Network, which learns the value of each action.
   - [PPO.md](PPO.md) and `ppo.py`: Proximal Policy Optimization, which learns the policy directly.
   - `rl.py`: the evaluation and metrics shared by both.
5. **The dashboard** (optional): `monitor.py` keeps a picture of the models while they train or run, `dashboard.py`
   serves it to the browser, and `dashboard.html` draws it (see "Watching a model live" below).
6. **The tests**: `test_feedforward.py` checks backpropagation against gradients measured numerically,
   `test_dqn.py` computes DQN and Double DQN targets on a tiny example you can follow by hand, and `test_ppo.py`
   checks the PPO gradient numerically and computes advantages by hand.

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

### 3. Handwritten digits: MNIST

A network with one hidden layer of 128 neurons learns to recognize handwritten digits in 28 x 28 images, 784 pixels
per image. It reaches about 97.7% on the 10,000 test images in 10 epochs. It shows a real dataset, the Adam optimizer
and validation. `mnist.py` downloads the dataset once (about 11 MB) into `datasets/mnist/` and reads its file format.

```sh
python train-mnist.py   # trains and saves models/mnist.joblib (about 15 seconds)
python run-mnist.py     # per-digit report, confusion matrix, and the first mistakes in mnist-errors.png
```

### 4. Reinforcement learning: CartPole and Acrobot

Two agents learn to balance a pole on a cart ([CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole)):
a [Deep Q-Network](DQN.md), which learns the value of each action, and [PPO](PPO.md), which learns the policy directly.

```sh
python train-cartpole.py                                  # DQN: saves the best model to models/cartpole.joblib
python train-cartpole-ppo.py                              # PPO: saves the best policy to models/cartpole-ppo.joblib
python run-agent.py CartPole-v1 models/cartpole.joblib    # watch an agent play (any of the two files)
python show-metrics.py metrics/cartpole.csv               # plots the training metrics to cartpole_<run>.png
```

PPO reaches the maximum of 500 in every run and stays there, while DQN averages about 420 to 450 and oscillates
("DQN or PPO?" in PPO.md compares them, with the limits of the comparison).
`finetune-cartpole.py` continues training the DQN model with less exploration, and saves the best model to
`models/cartpole-checkpoint.joblib` and the last one to `models/cartpole-finetuned.joblib`.

PPO also learns a harder environment, [Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot), a
two-link arm that has to swing its tip above a line, where DQN isn't reliable (see "Other environments" in DQN.md):

```sh
python train-acrobot-ppo.py                                  # about 2 minutes: saves models/acrobot-ppo.joblib
python run-agent.py Acrobot-v1 models/acrobot-ppo.joblib
python show-metrics.py metrics/acrobot-ppo.csv
```

Every script can be run from any directory, and `--help` shows its options. The training scripts take `--seed` to
make a run reproducible, `--epochs`, `--episodes` or `--steps` to change the length of the training, and, for the
environments that can be shown, `--render` to watch the training (much slower):

```sh
python train-cartpole.py --seed 0 --episodes 300
python train-acrobot-ppo.py --render
```

## Watching a model live

Every `train` and `run` script takes `--dashboard`, which opens a page at http://localhost:8000 (or the next free
port) that follows the model while it works:

```sh
python train-mnist.py --dashboard
python run-agent.py CartPole-v1 models/cartpole-ppo.joblib --dashboard
```

The page has three parts:

- **What the network does**: the input, the layers and the answer for one input, in plain words. For example, the
  image of a digit, how many neurons of the hidden layer it activates, and the digit the network answers with how
  sure it is.
- **Is it learning?**: the metrics the training reports, whether each one is improving, and what it means. It warns
  when the validation error rises while the training error keeps falling (overfitting).
- **Health checks**: weights that are no longer numbers, gradients that vanish or explode, ReLUs that never fire, and
  sigmoid or tanh outputs stuck at their limits.

"Inside the layers (advanced)" shows every value of the forward pass and every weight. The page only shows: it
doesn't change the model or the training. When the script ends, the page stays open with the last state until
Ctrl+C.

It works with any `FeedForward` model, whatever it does: the model tells its observers about each forward pass,
update and metric (`notify` in `feedforward.py`), and `monitor.py` keeps a copy at most every 0.1 s, so watching a
training barely slows it down (MNIST trains about 7% slower with the page open). The server only listens on this machine and uses only the standard library.
