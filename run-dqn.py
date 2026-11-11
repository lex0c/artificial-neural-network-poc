import sys

import gymnasium as gym
import numpy as np

from feedforward import load_model


# Watches a trained DQN agent play with the greedy policy (always the action with the highest Q-value).
#
#   python run-dqn.py CartPole-v1 models/cartpole.joblib
#   python run-dqn.py Acrobot-v1 models/acrobot.joblib
#   python run-dqn.py CartPole-v1 models/cartpole-checkpoint.joblib 3   # number of episodes (default 10)

if len(sys.argv) < 3:
    sys.exit("usage: python run-dqn.py <Gymnasium environment> <model file> [episodes]")

env_id, model_filepath = sys.argv[1], sys.argv[2]
episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10

model = load_model(model_filepath)
model.summary()

env = gym.make(env_id, render_mode="human")

for e in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        q_values = model.predict(state.reshape(1, -1))
        state, reward, terminated, truncated, _ = env.step(int(np.argmax(q_values)))
        total_reward += reward

        if terminated or truncated:
            break

    print(f"episode: {e+1}/{episodes} - reward: {total_reward:.0f}")

env.close()
