import argparse
import math
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from cli import display_path, project_path


# Plots the metrics that train_dqn and train_ppo write to a CSV file (metrics_filepath): one chart per column, against
# the episode (DQN) or the iteration (PPO), and one figure per training run (metrics_key), saved as
# <file name>_<metrics_key>.png. DQN.md and PPO.md explain how to read them.
#
#   python show-metrics.py metrics/cartpole.csv
#   python show-metrics.py metrics/acrobot-ppo.csv

parser = argparse.ArgumentParser(description="Plots the metrics of a DQN or PPO training.")
parser.add_argument("metrics", help="metrics file, e.g. metrics/cartpole.csv")
metrics_filepath = parser.parse_args().metrics

if not os.path.exists(metrics_filepath):
    sys.exit(f"{metrics_filepath} not found: the training scripts write their metrics to metrics/ (e.g. python train-cartpole.py)")

name = os.path.splitext(os.path.basename(metrics_filepath))[0]  # e.g. cartpole

data = pd.read_csv(metrics_filepath)
x = 'episode' if 'episode' in data.columns else 'iteration'
columns = [c for c in data.columns if c not in ('metrics_key', x) and pd.api.types.is_numeric_dtype(data[c])]

plt.style.use('ggplot')

for key in data['metrics_key'].unique():
    session_data = data[data['metrics_key'] == key]

    ncols = 3
    nrows = math.ceil(len(columns) / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows), squeeze=False)
    fig.suptitle(f'{name} - {key}', fontsize=16)

    for ax in axs.ravel()[len(columns):]:
        ax.axis('off')

    for ax, column in zip(axs.ravel(), columns):
        # Some columns only have values in some rows (e.g. greedy_reward, every eval_freq episodes)
        values = session_data[[x, column]].dropna()
        ax.plot(values[x], values[column], marker='o', markersize=3, linestyle='-')
        ax.set_title(column.replace('_', ' ').capitalize())
        ax.set_xlabel(x.capitalize())

    plt.tight_layout(pad=3.0)
    figure_filepath = project_path(f'{name}_{key}.png')
    plt.savefig(figure_filepath, dpi=150)
    plt.close(fig)
    print(f"Metrics of the run {key} saved to {display_path(figure_filepath)}")
