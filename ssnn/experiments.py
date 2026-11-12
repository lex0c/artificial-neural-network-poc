import time

import numpy as np

from ssnn.baselines import dense_mlp, hidden_for_parameters, mlp_parameters, train_mlp
from ssnn.metrics import accuracy, accuracy_by_group, routing_summary
from ssnn.model import SSNN, parameter_counts
from ssnn.training import cross_entropy, evaluate, train


# Training and testing one model on a classification task given as (train, validation, test) splits, each
# (inputs, labels, groups), where the groups are the regimes. The result has the same keys for every model, so the
# scripts can print and record them the same way.


def ssnn_result(name, config, data):
    train_set, val_set, test_set = data
    model = SSNN(config)
    start = time.time()
    history = train(model, train_set, val_set)
    seconds = time.time() - start

    probabilities, selected = evaluate(model, test_set[0])
    routing = routing_summary(selected, config["num_experts"], test_set[2])
    counts = parameter_counts(config)
    return {**scores(name, probabilities, test_set, history, seconds), "parameters": counts["total"],
            "per_event": counts["per_event"], "unused_experts": routing["unused"],
            "usage_entropy": routing["usage_entropy"], "regime_information": routing["group_information"]}


# A dense MLP with hidden neurons, or with as many parameters as target_parameters
def mlp_result(name, config, data, seed, target_parameters=None, hidden=None):
    train_set, val_set, test_set = data
    hidden = hidden or hidden_for_parameters(target_parameters, config["input_dim"], config["num_classes"])
    model = dense_mlp(config["input_dim"], hidden, config["num_classes"], seed)
    start = time.time()
    history = train_mlp(model, train_set, val_set, config)
    seconds = time.time() - start

    count = mlp_parameters(config["input_dim"], hidden, config["num_classes"])
    return {**scores(f"{name} ({hidden} hidden)", model.predict(test_set[0]), test_set, history, seconds),
            "parameters": count, "per_event": count, "unused_experts": "", "usage_entropy": "", "regime_information": ""}


def scores(name, probabilities, test_set, history, seconds):
    predictions = probabilities.argmax(axis=1)
    by_regime = accuracy_by_group(predictions, test_set[1], test_set[2])
    return {"model": name, "best_epoch": history["best_epoch"], "val_loss": min(history["val_loss"]),
            "test_loss": cross_entropy(probabilities, test_set[1]), "test_accuracy": accuracy(predictions, test_set[1]),
            "by_regime": by_regime, "worst_regime": min(by_regime.values()), "seconds": seconds}


def print_header(regimes):
    columns = "regime 0, 1, 2, 3" if regimes <= 4 else "worst regime"
    print(f"  {'model':33s} {'params':>8} {'/event':>7}   {'test':>6}   {columns:28s} routing")


def print_row(result):
    by_regime = result["by_regime"]
    regimes = (" ".join(f"{value * 100:5.1f}" for value in by_regime.values()) if len(by_regime) <= 4
               else f"{result['worst_regime'] * 100:5.1f}")
    routing = (f"{result['unused_experts']:>2} unused, {result['regime_information']:.2f} bits"
               if result["unused_experts"] != "" else "")
    print(f"  {result['model']:33s} {result['parameters']:>8,} {result['per_event']:>7,}   "
          f"{result['test_accuracy'] * 100:5.1f}%   {regimes:28s} {routing:24s} {result['seconds']:5.0f}s")


def print_means(results):
    for name in dict.fromkeys(result["model"] for result in results):
        values = [result["test_accuracy"] for result in results if result["model"] == name]
        print(f"  {name:33s} {np.mean(values) * 100:5.1f}% (from {min(values) * 100:.1f} to {max(values) * 100:.1f})")
