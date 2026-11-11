# Neural Network POC

This framework is designed for educational and prototyping purposes.

- [FeedForward](/FEEDFORWARD.md)

## Run

Requires Python 3.12 or newer.

```sh
pip install -r requirements.txt
```

### simple_linear_function

`Y = 2X - 1`

Predict

```sh
python run.py
```

Train

```sh
python train.py
```

### [cartpole](https://gymnasium.farama.org/environments/classic_control/cart_pole)

Implementation of a Deep Q-Network (DQN) agent. The training loop lives in `dqn.py`; pass `double_dqn=True` to `train_dqn` to use Double DQN.

Every 50 episodes the training evaluates the greedy policy (no exploration) over 10 episodes, and the model with the best evaluation is saved: the DQN policy often gets worse after reaching its best, so the last model isn't necessarily the best one. The metrics in `metrics/cartpole.csv` include the greedy reward and the mean Q-value, which should level off near `1 / (1 - gamma)` when the agent keeps the pole up.

Run the trained model

```sh
python run-cartpole.py
```

Fine-tuning the model
```sh
python finetune-cartpole.py
```

Train model
```sh
python train-cartpole.py
```

Show training metrics
```sh
python show-cartpole-metrics.py
```

## Tests

Checks the backpropagation against numerical gradients, among other things. They also run on GitHub Actions for every pull request:

```sh
python -m unittest -v
```
