import argparse
import os
import sys

from feedforward import load_model
from progress import display_path


# Helpers for the scripts (training, running and plotting): their command-line options, the paths of the project and
# clear error messages. The machine learning code doesn't depend on this module.

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# The path of a file of the project (models/, metrics/, datasets/...), so the scripts work from any directory.
def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)


# The options of the training scripts: --seed, --render (for the environments that can be shown) and the length of
# the training (e.g. episodes=500 adds --episodes, 500 by default). Every script also has --help.
def training_args(description, render=False, **length):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the random numbers, to make the run reproducible (default: a different run each time)")

    if render:
        parser.add_argument("--render", action="store_true", help="show the training episodes in a window (much slower)")

    for name, default in length.items():
        parser.add_argument(f"--{name}", type=int, default=default, help=f"number of {name} (default: {default})")

    return parser.parse_args()


# Loads a model, or explains how to create it when the file doesn't exist.
def load_trained_model(path, train_command):
    if not os.path.exists(path):
        sys.exit(f"{display_path(path)} not found: train it first with python {train_command}")

    return load_model(path)
