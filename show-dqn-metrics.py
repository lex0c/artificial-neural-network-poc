import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


# Plots the metrics that train_dqn writes to a CSV file (metrics_filepath), one figure per training run (metrics_key),
# saved as <file name>_<metrics_key>.png. DQN.md explains how to read them.
#
#   python show-dqn-metrics.py metrics/cartpole.csv
#   python show-dqn-metrics.py metrics/acrobot.csv

if len(sys.argv) < 2:
    sys.exit("usage: python show-dqn-metrics.py <metrics file>")

metrics_filepath = sys.argv[1]
name = os.path.splitext(os.path.basename(metrics_filepath))[0]  # e.g. cartpole

data = pd.read_csv(metrics_filepath)
unique_keys = data['metrics_key'].unique()

plt.style.use('ggplot')

for key in unique_keys:
    # Filter data for the current metrics_key
    session_data = data[data['metrics_key'] == key]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
    fig.suptitle(f'{name} - {key}', fontsize=16)

    # Total reward per episode
    axs[0, 0].plot(session_data['episode'], session_data['total_reward'], marker='o', linestyle='-', color='b')
    axs[0, 0].set_title('Total Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')

    # Average rolling reward per episode
    axs[0, 1].plot(session_data['episode'], session_data['average_rolling_reward'], marker='o', linestyle='-', color='r')
    axs[0, 1].set_title('Average Rolling Reward per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Average Rolling Reward')

    # Loss per episode (if available)
    if 'loss' in session_data.columns and session_data['loss'].notna().any():
        axs[0, 2].plot(session_data['episode'], session_data['loss'], marker='o', linestyle='-', color='g')
        axs[0, 2].set_title('Loss per Episode')
        axs[0, 2].set_xlabel('Episode')
        axs[0, 2].set_ylabel('Loss')

    # Duration per episode
    axs[1, 0].plot(session_data['episode'], session_data['duration'], marker='o', linestyle='-', color='m')
    axs[1, 0].set_title('Duration per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Duration (s)')

    # Epsilon per episode
    axs[1, 1].plot(session_data['episode'], session_data['epsilon'], marker='o', linestyle='-', color='c')
    axs[1, 1].set_title('Epsilon Over Episodes')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon')

    # Steps per episode
    axs[1, 2].plot(session_data['episode'], session_data['steps'], marker='o', linestyle='-', color='y')
    axs[1, 2].set_title('Steps per Episode')
    axs[1, 2].set_xlabel('Episode')
    axs[1, 2].set_ylabel('Steps')

    # Learning rate per episode
    axs[2, 0].plot(session_data['episode'], session_data['learning_rate'], marker='o', linestyle='-', color='purple')
    axs[2, 0].set_title('Learning Rate Over Episodes')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Learning Rate')

    # Greedy evaluations: only the episodes where one ran (every eval_freq episodes)
    if 'greedy_reward' in session_data.columns and session_data['greedy_reward'].notna().any():
        evaluations = session_data[session_data['greedy_reward'].notna()]
        axs[2, 1].plot(evaluations['episode'], evaluations['greedy_reward'], marker='o', linestyle='-', color='darkorange')
        axs[2, 1].set_title('Greedy Reward (no exploration)')
        axs[2, 1].set_xlabel('Episode')
        axs[2, 1].set_ylabel('Mean Reward')

    # Mean of the highest Q-value of the sampled states (empty until the replay memory is large enough)
    if 'mean_q' in session_data.columns and session_data['mean_q'].notna().any():
        axs[2, 2].plot(session_data['episode'], session_data['mean_q'], marker='o', linestyle='-', color='teal')
        axs[2, 2].set_title('Mean Q-Value per Episode')
        axs[2, 2].set_xlabel('Episode')
        axs[2, 2].set_ylabel('Mean Max Q')

    plt.tight_layout(pad=3.0)
    plt.savefig(f'{name}_{key}.png', dpi=300)
    plt.close(fig)

