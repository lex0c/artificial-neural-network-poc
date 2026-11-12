# Neural Network POC

A small neural network framework written with NumPy only, to learn how neural networks and Deep Q-Learning work.
Every step, from the forward pass to backpropagation and the DQN training loop, is plain code meant to be read.

## Setup

Requires Python 3.12 or newer.

```sh
pip install -r requirements.txt
python -m unittest -v   # optional: checks that everything works
```

LunarLander (example 4) also needs [Box2D](https://gymnasium.farama.org/environments/box2d/), compiled with SWIG. On
Python 3.14, in the activated virtual environment:

```sh
pip install swig setuptools
pip install --no-build-isolation box2d-py==2.3.8
```

On Python 3.13 or older, Gymnasium installs another Box2D package with `pip install "gymnasium[box2d]"` (not tested
here).

## Reading order

1. **[FEEDFORWARD.md](FEEDFORWARD.md)**: how a network computes and learns (forward pass, loss, backpropagation,
   gradient descent), with one training step worked out by hand.
2. **The framework**:
   - `feedforward.py`: the `FeedForward` model. Start with `train_step`, which calls the five steps of a training step.
   - `activations.py`: ReLU, sigmoid, tanh, linear and softmax, and their derivatives.
   - `losses.py`: mean squared error and cross-entropy (categorical and binary), and their derivatives.
   - `optimizers.py`: plain gradient descent (SGD), momentum, RMSprop and Adam.
   - `data.py`: one-hot encoding, normalization, batches and seeds.
   - `gradcheck.py`: gradients measured numerically, to check the ones written by hand.
3. **The examples**, from the simplest (below).
4. **Reinforcement learning**:
   - [DQN.md](DQN.md) and `dqn.py`: a Deep Q-Network, which learns the value of each action.
   - [PPO.md](PPO.md) and `ppo.py`: Proximal Policy Optimization, which learns the policy directly.
   - `rl.py`: the evaluation and metrics shared by both.
5. **Watching** (optional): `monitor.py` keeps a picture of the models while they train or run, `dashboard.py`
   serves it to the browser, and `dashboard.html` draws it (see "Watching a model live" below). `viewer.py` shows an
   agent playing in a window while it trains (`--watch`).
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

The training of 10 epochs stops near the best moment: after it, the validation loss rises while the training loss
keeps falling (overfitting). `python train-mnist.py --epochs 30 --patience 3` finds that moment by itself (early
stopping): it stops when the validation loss hasn't improved for 3 epochs and goes back to the best epoch. Over 5
seeds it stopped after 8 to 12 epochs and scored the same 97.8% as the 10 fixed epochs, which were chosen by hand.
`python train-mnist.py --epochs 20 --dropout 0.2` (dropout, see FEEDFORWARD.md) scored 98.0% over 5 seeds, every
one of them better than every seed of the default.

### 4. Reinforcement learning: CartPole, Acrobot and LunarLander

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

And it lands a spacecraft in [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/), which needs
Box2D (see Setup). The lander has to touch down gently between two flags, firing its main engine or one of its two
side engines, and 200 points count as solved. Over 9 seeds, every run solves it in about 4 minutes, and the saved
policies score 272 on average on 100 new episodes, and 200 or more in 99% of them.

```sh
python train-lunarlander-ppo.py --watch                          # about 4 minutes: saves models/lunarlander-ppo.joblib
python run-agent.py LunarLander-v3 models/lunarlander-ppo.joblib
python show-metrics.py metrics/lunarlander-ppo.csv
```

`--watch` opens a window where the newest policy plays while it trains, without slowing the training: a separate
process plays one episode after the other, each with the weights of the last iteration (`viewer.py`). Rendering the
training itself (`--render` in the other scripts) would hold it to the 50 steps per second of the animation.

Every script can be run from any directory, and `--help` shows its options. The training scripts take `--seed` to
make a run reproducible, `--epochs`, `--episodes` or `--steps` to change the length of the training, and, for the
environments that can be shown, `--render` to watch the training (much slower) or, for LunarLander, `--watch`:

```sh
python train-cartpole.py --seed 0 --episodes 300
python train-acrobot-ppo.py --render
```

### 5. Decision boundaries: two spirals

Two spirals wind around each other, one per class, and two models learn to tell them apart: one without hidden
layers, whose boundary can only be a straight line (62% to 67% right over 5 seeds), and one with two hidden layers of
16 neurons, which bends its boundary to follow the arms (99.1% to 99.8%). `spirals.png` shows both boundaries, and the network's one
taking shape along the training. It also shows binary cross-entropy, with one sigmoid output.

```sh
python train-spirals.py   # about 3 seconds: saves spirals.png
```

### 6. An autoencoder

A network learns to rebuild MNIST digits through a layer of only 2 numbers. It never sees the labels, yet the 2
numbers of each digit gather by digit, and `autoencoder.png` shows them as points, with the reconstructions and what
the second half of the network draws for each point of the plane.

```sh
python train-autoencoder.py   # about 1 minute: saves autoencoder.png
```

### 7. Comparing optimizers

SGD, momentum, RMSprop and Adam train the same network over a range of learning rates (see "Optimizers" in
FEEDFORWARD.md for the results). `optimizers.png` shows the loss for each learning rate, and the training of each
optimizer with its best one.

```sh
python compare-optimizers.py   # about 40 seconds: saves optimizers.png
```

### 8. Overfitting and regularization

With few samples, a network learns them too well: it memorizes their noise, and does worse on new data than on the
data it learned (overfitting). Two examples show it, and what early stopping, dropout and L2 do about it, each over
20 seeds.

`train-breast-cancer.py` asks a real yes/no question: is a breast tumor malignant? The 569 tumors of the breast
cancer dataset of scikit-learn (no download) have 30 measurements each, and the same network trains in three ways.
Without regularization the validation loss rises again after about 100 epochs, while the training loss keeps
falling; `breast-cancer.png` shows the curves.

| | Test accuracy | Test loss |
| --- | --- | --- |
| No regularization, 500 epochs | 96.5% | 0.156 |
| Early stopping (patience 20), stopping after about 170 epochs | 97.0% | 0.093 |
| Dropout 0.5, 500 epochs | 97.0% | 0.130 |

The overfitting shows more in the loss than in the accuracy: the longer the training, the surer the model is of its
wrong answers.

`compare-regularization.py` trains a large network on only 100 noisy points of each spiral, and `regularization.png`
draws each boundary: without regularization it bends around every point, even those the noise pushed into the other
spiral.

| | Right on the training points | Right on new points |
| --- | --- | --- |
| No regularization | 99.0% | 87.0% |
| L2 0.0003 | 97.2% | 88.1% |
| Dropout 0.2 | 95.9% | 88.3% |
| Early stopping (patience 50) | 94.2% | 87.9% |
| Too much: L2 0.003 | 72.1% | 68.8% |

Each one narrows the gap between the two columns, but the last line shows the other side: too much regularization
keeps the network from following the spirals at all (underfitting).

```sh
python train-breast-cancer.py      # about 6 seconds: saves breast-cancer.png
python compare-regularization.py   # about 15 seconds: saves regularization.png
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
