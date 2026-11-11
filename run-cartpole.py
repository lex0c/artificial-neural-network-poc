import sys

import gymnasium as gym
import numpy as np

from feedforward import load_model


# Watches a trained agent play CartPole with the greedy policy (always the action with the highest Q-value).
#
#   python run-cartpole.py                                    # the model saved by train-cartpole.py
#   python run-cartpole.py models/cartpole-checkpoint.joblib  # the model saved by finetune-cartpole.py

model_filepath = sys.argv[1] if len(sys.argv) > 1 else 'models/cartpole.joblib'
model = load_model(model_filepath)
model.summary()

env = gym.make("CartPole-v1", render_mode="human")
episodes = 30

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
