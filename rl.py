import csv

import numpy as np


# Helpers shared by the reinforcement learning agents (dqn.py and ppo.py).


# Plays episodes with the greedy policy and returns the mean total reward: what the agent has learned, without the
# exploration of the training. The greedy action is the one with the highest output of the model: the highest Q-value
# of a DQN, or the most likely action of a PPO policy. Each episode starts from a fixed seed, so every evaluation along
# the training uses the same starting states.
def evaluate_greedy(env, model, episodes=10, seed=0):
    num_observations = env.observation_space.shape[0]
    total_rewards = []

    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        total_reward = 0

        while True:
            outputs = model.predict(np.reshape(state, [1, num_observations]))
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(outputs)))
            total_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(total_reward)

    return float(np.mean(total_rewards))


# Adds a row of metrics to a CSV file (see show-metrics.py)
def write_to_csv(filepath, metric):
    fieldnames = list(metric.keys())

    with open(filepath, 'a+', newline='') as file:
        file.seek(0)
        header = file.readline().strip()

        # Rows with other columns under the existing header would misalign the whole file
        if header and header != ','.join(fieldnames):
            raise ValueError(f"{filepath} has other columns ({header}); move it away to start a new metrics file")

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not header:
            writer.writeheader()

        writer.writerow(metric)
