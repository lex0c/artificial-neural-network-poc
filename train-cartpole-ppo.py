import gymnasium as gym

from data import set_seed
from feedforward import FeedForward
from ppo import train_ppo


# Trains a PPO agent on CartPole (see PPO.md) and saves the best policy to models/cartpole-ppo.joblib, to compare
# with the DQN agent of train-cartpole.py.

SEED = None     # set an integer (e.g. 0) to make the run reproducible
RENDER = False  # True to watch the training episodes (much slower)

env = gym.make("CartPole-v1", render_mode="human" if RENDER else None)
set_seed(SEED, env)
eval_env = gym.make("CartPole-v1")  # a separate environment for the evaluations


def network(num_outputs, output_act_fn, loss):
    model = FeedForward()
    model.add_layer(num_inputs=env.observation_space.shape[0], num_neurons=64, act_fn='tanh')
    model.add_layer(num_inputs=64, num_neurons=64, act_fn='tanh')
    model.add_layer(num_inputs=64, num_neurons=num_outputs, act_fn=output_act_fn)
    model.configure(loss=loss, optimizer='adam')
    return model


# The 4 numbers of the state in; the policy gives one probability per action (push left or right), and the value
# model the return expected from the state
policy = network(env.action_space.n, 'softmax', None)  # its gradient comes from the PPO objective
value_model = network(1, 'linear', 'mse')
policy.summary()


train_ppo(
    env, policy, value_model,
    total_steps=100000,
    learning_rate=0.0003,
    num_steps=2048,          # steps played with the current policy in each iteration...
    num_epochs=10,           # ...then 10 passes over them...
    minibatch_size=64,       # ...in minibatches of 64
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,          # each update can change the probability of an action by about 20% at most
    entropy_coef=0.01,       # a bonus for keeping some randomness, so the policy doesn't stop exploring too early
    eval_env=eval_env,
    eval_freq=5,             # iterations between evaluations
    eval_episodes=10,
    metrics_filepath='metrics/cartpole-ppo.csv',  # plotted by show-metrics.py
    checkpoint_filepath='models/cartpole-ppo.joblib',  # keeps the policy with the best evaluation
)


env.close()
eval_env.close()
