import argparse
import sys

import gymnasium as gym
import numpy as np

from cli import add_dashboard_option, keep_dashboard_open, load_trained_model, start_dashboard


# Watches a trained agent play with the greedy policy: the action with the highest output of the model, which is the
# highest Q-value of a DQN or the most likely action of a PPO policy.
#
#   python run-agent.py CartPole-v1 models/cartpole.joblib                 # DQN
#   python run-agent.py Acrobot-v1 models/acrobot-ppo.joblib               # PPO
#   python run-agent.py CartPole-v1 models/cartpole-ppo.joblib --episodes 3

parser = argparse.ArgumentParser(description="Watches a trained agent play with the greedy policy.")
parser.add_argument("environment", help="Gymnasium environment, e.g. CartPole-v1 or Acrobot-v1")
parser.add_argument("model", help="model file, e.g. models/cartpole.joblib")
parser.add_argument("--episodes", type=int, default=10, help="number of episodes (default: 10)")
add_dashboard_option(parser)
args = parser.parse_args()

model = load_trained_model(args.model, "train-cartpole.py, or another train-*.py script")
env = gym.make(args.environment, render_mode="human")

# The model must have been trained for this environment: one input per number of the state, one output per action
num_inputs, num_outputs = model.layers[0]["weights"].shape[1], model.layers[-1]["weights"].shape[0]
state_size, num_actions = env.observation_space.shape[0], env.action_space.n

if (num_inputs, num_outputs) != (state_size, num_actions):
    sys.exit(f"{args.model} has {num_inputs} inputs and {num_outputs} outputs, but {args.environment} has a state of "
             f"{state_size} numbers and {num_actions} actions: it was trained for another environment")

model.summary()
monitor = start_dashboard(args.dashboard, {"agent": model})

for e in range(args.episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        outputs = model.predict(state.reshape(1, -1))
        state, reward, terminated, truncated, _ = env.step(int(np.argmax(outputs)))
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Episode {e+1}/{args.episodes} - reward {total_reward:.0f}")
    model.notify("on_metrics", {"episode": e + 1, "reward": total_reward})

env.close()
keep_dashboard_open(monitor)
