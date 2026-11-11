import argparse
import os
import sys
import time

from feedforward import load_model
from progress import display_path


# Helpers for the scripts (training, running and plotting): their command-line options, the paths of the project and
# clear error messages. The machine learning code doesn't depend on this module.

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# The path of a file of the project (models/, metrics/, datasets/...), so the scripts work from any directory.
def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)


# The options of the training scripts: --seed, --render (for the environments that can be shown), --watch (for the
# scripts that use viewer.py) and the length of the training (e.g. episodes=500 adds --episodes, 500 by default).
# Every script also has --help.
def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def training_args(description, render=False, watch=False, **length):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the random numbers, to make the run reproducible (default: a different run each time)")

    if render:
        parser.add_argument("--render", action="store_true", help="show the training episodes in a window (much slower)")

    if watch:
        parser.add_argument("--watch", action="store_true",
                            help="show the newest policy playing in a window while it trains, without slowing the training")

    for name, default in length.items():
        parser.add_argument(f"--{name}", type=positive_int, default=default, help=f"number of {name} (default: {default})")

    add_dashboard_option(parser)
    return parser.parse_args()


# The options of the scripts that use a trained model: --dashboard (and --help)
def run_args(description):
    parser = argparse.ArgumentParser(description=description)
    add_dashboard_option(parser)
    return parser.parse_args()


def add_dashboard_option(parser):
    parser.add_argument("--dashboard", action="store_true",
                        help="show the model live in the browser: its layers, weights, activations and metrics")


# With --dashboard, starts the dashboard (dashboard.py) for the models given as {name: model} and returns its monitor;
# otherwise returns None and changes nothing.
def start_dashboard(enabled, models):
    if not enabled:
        return None

    from dashboard import Dashboard
    from monitor import Monitor

    monitor = Monitor()
    for name, model in models.items():
        monitor.attach(model, name)

    print(f"Dashboard: {Dashboard(monitor).start()} (open it in the browser)")
    return monitor


# Keeps the dashboard open when the script is done, to explore its last state, until Ctrl+C.
def keep_dashboard_open(monitor):
    if monitor is None:
        return

    print("The dashboard stays open with the last state: press Ctrl+C to exit")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()


# Loads a model, or explains how to create it when the file doesn't exist.
def load_trained_model(path, train_command):
    if not os.path.exists(path):
        sys.exit(f"{display_path(path)} not found: train it first with python {train_command}")

    return load_model(path)
