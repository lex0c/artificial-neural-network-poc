import gymnasium as gym

from cli import keep_dashboard_open, start_dashboard, project_path, training_args
from data import set_seed
from dqn import train_dqn
from feedforward import FeedForward, clone_model


# Trains a DQN agent on CartPole (see DQN.md) and saves the best model to models/cartpole.joblib.

args = training_args("Trains a DQN agent on CartPole and saves the best model to models/cartpole.joblib.", render=True, episodes=500)

env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
set_seed(args.seed, env)
eval_env = gym.make("CartPole-v1")  # a separate environment for the greedy evaluations


# Q-network: the 4 numbers of the state in, one Q-value per action (push left, push right) out
model = FeedForward()
model.add_layer(num_inputs=env.observation_space.shape[0], num_neurons=64, act_fn='relu')
model.add_layer(num_inputs=64, num_neurons=env.action_space.n, act_fn='linear')  # a Q-value can be any number
model.configure(loss="mse")
model.summary()
monitor = start_dashboard(args.dashboard, {"Q-network": model})

target_model = clone_model(model)  # gives stable Bellman targets (see DQN.md)


train_dqn(
    env, model, target_model,
    episodes=args.episodes,
    batch_size=512,          # transitions sampled from the replay memory after each episode
    gamma=0.80,              # discount factor: a reward k steps ahead counts gamma^k (0.33 after 5 steps with 0.8)
    learning_rate=0.001,
    epsilon=1.0,             # start with random actions only...
    epsilon_min=0.1,         # ...down to 10% of random actions
    epsilon_decay=0.990,     # epsilon is multiplied by this after every episode
    target_update_freq=10,   # episodes between copies of the model into the target model
    double_dqn=False,        # True for Double DQN, which reduces the overestimation of Q-values
    eval_env=eval_env,
    eval_freq=50,            # episodes between greedy evaluations
    eval_episodes=10,
    metrics_filepath=project_path('metrics', 'cartpole.csv'),  # plotted by show-metrics.py
    checkpoint_filepath=project_path('models', 'cartpole.joblib'),  # keeps the model with the best greedy evaluation
)


env.close()
eval_env.close()
keep_dashboard_open(monitor)
