import argparse
import os
import sys
import time

from progress import display_path


# Helpers for the scripts (training, running and plotting): their command-line options, the paths of the project and
# clear error messages. The machine learning code doesn't depend on this module.

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# The path of a file of the project (models/, metrics/, datasets/...), so the scripts work from any directory.
def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)


# An argparse type for the options that count something (epochs, episodes, steps): at least 1
def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


# An argparse type for a fraction of the inputs to drop: at least 0 and smaller than 1
def dropout_rate(text):
    value = float(text)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 0 and smaller than 1, got {value}")
    return value


# The options of the training scripts: --seed, --render (for the environments that can be shown), --watch (for the
# scripts that use viewer.py), --patience (for the ones with validation data), --dropout, the length of the training
# (e.g. episodes=500 adds --episodes, 500 by default) and --dashboard (unless dashboard=False). Every script also has
# --help.
def training_args(description, render=False, watch=False, patience=False, dropout=False, dashboard=True, **length):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the random numbers, to make the run reproducible (default: a different run each time)")

    if render:
        parser.add_argument("--render", action="store_true", help="show the training episodes in a window (much slower)")

    if watch:
        parser.add_argument("--watch", action="store_true",
                            help="show the newest policy playing in a window while it trains, without slowing the training")

    if patience:
        parser.add_argument("--patience", type=positive_int, default=None,
                            help="early stopping: stop when the validation loss hasn't improved for this many epochs, "
                                 "and keep the weights of the best epoch (default: train all the epochs)")

    if dropout:
        parser.add_argument("--dropout", type=dropout_rate, default=0.0,
                            help="fraction of the inputs of the output layer to drop in each training step (default: 0)")

    for name, default in length.items():
        parser.add_argument(f"--{name}", type=positive_int, default=default, help=f"number of {name} (default: {default})")

    if dashboard:
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

    monitor.refresh()  # the final weights, even when the last copy came before them
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

    from feedforward import load_model  # here, so importing cli doesn't load feedforward (and its dependencies)

    return load_model(path)
