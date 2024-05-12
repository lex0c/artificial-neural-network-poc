# Neural Network POC

This framework is designed for educational and prototyping purposes.

- [FeedForward](/FEEDFORWARD.md)

## Run

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

Implementation of a Deep Q-Network (DQN) agent.

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

