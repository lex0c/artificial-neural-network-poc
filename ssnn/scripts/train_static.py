import argparse
import csv
import os
import platform
import subprocess
import time
from datetime import datetime

import numpy as np

from cli import positive_int
from ssnn.baselines import dense_mlp, hidden_for_parameters, mlp_parameters, train_mlp
from ssnn.config import config_hash, make_config
from ssnn.metrics import accuracy, accuracy_by_group, routing_summary
from ssnn.model import SSNN, parameter_counts
from ssnn.tasks import experiment_a
from ssnn.training import cross_entropy, evaluate, train


# Experiment A: classification with 4 regimes (ssnn/tasks.py), for each seed:
#   ssnn          the SSNN, learned router, k = 2 of 16 experts
#   random router the same with k experts drawn at random per event (does the router learn something useful? H5)
#   all experts   the same with k = 16 (does selecting few experts help or hurt?)
#   mlp total     a dense MLP with as many parameters as the SSNN has in total
#   mlp per event a dense MLP with as many parameters as the SSNN uses per event
#   mlp 64 hidden a dense MLP with as many hidden neurons as the k experts have (2 x 32): in this task every example
#                 starts from the zero state, so the recurrent weights W_h of the experts count as used per event but
#                 do nothing, and this is the comparison by the work that matters
# Every model trains with the same data, epochs, minibatches and learning rate, and keeps its epoch with the lowest
# validation loss. The test set is only used at the end. One row per model and seed goes to ssnn/reports/experiments.csv.
#
#   python -m ssnn.scripts.train_static --seeds 7 19 43

REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "experiments.csv")

parser = argparse.ArgumentParser(description="Experiment A of the SSNN: classification with 4 regimes.")
parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43], help="seeds of the data and models (default: 7 19 43)")
parser.add_argument("--epochs", type=positive_int, default=30, help="epochs of every model (default: 30)")
parser.add_argument("--examples", type=positive_int, nargs=3, default=[12000, 3000, 3000],
                    metavar=("TRAIN", "VAL", "TEST"), help="examples of each split (default: 12000 3000 3000)")
parser.add_argument("--report", default=REPORT, help="CSV file the results are added to")
args = parser.parse_args()


# The commit of the code, with -dirty when any file has changes that aren't committed, new files included
def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return commit + ("-dirty" if changes.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


ENVIRONMENT = {
    "commit": git_commit(), "numpy": np.__version__, "python": platform.python_version(), "machine": platform.machine(),
    "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
}


def report(row):
    new_file = not os.path.exists(args.report)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def ssnn_result(name, config, data):
    train_set, val_set, test_set = data
    model = SSNN(config)
    start = time.time()
    history = train(model, train_set, val_set)
    seconds = time.time() - start

    probabilities, selected = evaluate(model, test_set[0])
    predictions = probabilities.argmax(axis=1)
    routing = routing_summary(selected, config["num_experts"], test_set[2])
    counts = parameter_counts(config)
    return {"model": name, "parameters": counts["total"], "per_event": counts["per_event"],
            "best_epoch": history["best_epoch"], "val_loss": min(history["val_loss"]),
            "test_loss": cross_entropy(probabilities, test_set[1]), "test_accuracy": accuracy(predictions, test_set[1]),
            "by_regime": accuracy_by_group(predictions, test_set[1], test_set[2]), "unused_experts": routing["unused"],
            "usage_entropy": routing["usage_entropy"], "regime_information": routing["group_information"], "seconds": seconds}


def mlp_result(name, config, data, seed, target_parameters=None, hidden=None):
    train_set, val_set, test_set = data
    hidden = hidden or hidden_for_parameters(target_parameters, config["input_dim"], config["num_classes"])
    model = dense_mlp(config["input_dim"], hidden, config["num_classes"], seed)
    start = time.time()
    history = train_mlp(model, train_set, val_set, config)
    seconds = time.time() - start

    probabilities = model.predict(test_set[0])
    predictions = probabilities.argmax(axis=1)
    count = mlp_parameters(config["input_dim"], hidden, config["num_classes"])
    return {"model": f"{name} ({hidden} hidden)", "parameters": count, "per_event": count,
            "best_epoch": history["best_epoch"], "val_loss": min(history["val_loss"]),
            "test_loss": cross_entropy(probabilities, test_set[1]), "test_accuracy": accuracy(predictions, test_set[1]),
            "by_regime": accuracy_by_group(predictions, test_set[1], test_set[2]), "unused_experts": "",
            "usage_entropy": "", "regime_information": "", "seconds": seconds}


def print_row(result):
    regimes = " ".join(f"{value * 100:5.1f}" for value in result["by_regime"].values())
    routing = (f"{result['unused_experts']:>2} unused, {result['regime_information']:.2f} bits"
               if result["unused_experts"] != "" else "")
    print(f"  {result['model']:33s} {result['parameters']:>7,} {result['per_event']:>7,}   "
          f"{result['test_accuracy'] * 100:5.1f}%   {regimes}   {routing:24s} {result['seconds']:5.0f}s")


all_results = []

for seed in args.seeds:
    data = experiment_a(seed, sizes=tuple(args.examples))
    base = make_config(seed=seed, epochs=args.epochs)
    print(f"\nSeed {seed}: {args.examples[0]:,} examples to train, {args.examples[1]:,} to validate, {args.examples[2]:,} to test")
    print(f"  {'model':33s} {'params':>7} {'/event':>7}   {'test':>6}   regime 0, 1, 2, 3              routing")

    results = [
        ssnn_result("ssnn", base, data),
        ssnn_result("ssnn, random router", make_config(seed=seed, epochs=args.epochs, router="random"), data),
        ssnn_result("ssnn, all 16 experts", make_config(seed=seed, epochs=args.epochs, top_k=base["num_experts"]), data),
    ]
    counts = parameter_counts(base)
    results += [mlp_result("mlp, same total", base, data, seed, target_parameters=counts["total"]),
                mlp_result("mlp, same per event", base, data, seed, target_parameters=counts["per_event"]),
                mlp_result("mlp, same hidden neurons", base, data, seed, hidden=base["top_k"] * base["state_dim"])]

    for result in results:
        print_row(result)
        report({"date": datetime.now().isoformat(timespec="seconds"), "experiment": "A", "seed": seed,
                "config": config_hash(base), "epochs": args.epochs, "examples": "/".join(map(str, args.examples)),
                **{key: value for key, value in result.items() if key != "by_regime"},
                **{f"regime_{regime}": value for regime, value in result["by_regime"].items()}, **ENVIRONMENT})
    all_results += results

if len(args.seeds) > 1:
    print(f"\nMean over {len(args.seeds)} seeds (test accuracy):")
    for name in dict.fromkeys(result["model"] for result in all_results):
        values = [result["test_accuracy"] for result in all_results if result["model"] == name]
        print(f"  {name:33s} {np.mean(values) * 100:5.1f}% (from {min(values) * 100:.1f} to {max(values) * 100:.1f})")

print(f"\nResults added to {os.path.relpath(args.report)}")
