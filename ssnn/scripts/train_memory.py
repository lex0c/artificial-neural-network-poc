import argparse
import os
import time
from datetime import datetime

import numpy as np

from cli import positive_int
from ssnn.config import config_hash, make_config
from ssnn.metrics import memory_leak
from ssnn.model import SSNN
from ssnn.recurrent import GRU, SequenceSSNN, TanhRNN, gru_hidden_for
from ssnn.reporting import REPORTS, append_row, environment
from ssnn.tasks import MemoryTask, experiment_memory
from ssnn.training import evaluate_sequences, train_sequences


# Experiment B: memory. Registers of one of 2 values (or --values), a long gap of noisy NOOPs with rewrites of other
# registers, and a QUERY of one register at the end (ssnn/tasks.py). Trained with distances up to 30 (or
# --max-distance), tested up to 500. For each seed:
#   ssnn, oracle        one expert per register and one for the NOOPs, chosen by hand: what the architecture can do
#                       when the routing is right
#   ssnn, learned       16 experts (or --experts), 2 per event, the router learns which
#   ssnn, gated         the same, each expert writing in proportion to its weight (write "gated"), so the router can
#                       learn to protect a memory, with and without the balance loss
#   ssnn, dense         the gated one, with a router that also learns which experts to choose (router_gradient
#                       "dense", ssnn/gradients.py)
#   rnn                 a dense tanh RNN with 32 neurons, as much work per event as one expert
#   gru                 gated recurrent units, 32 neurons, the same with as much work per event as the learned SSNN,
#                       and 64 neurons
# Every model trains on the same sequences with the same epochs, minibatches and learning rate (each of
# --learning-rates, from the start), and keeps its epoch with the lowest validation loss. One row per model, learning
# rate and seed goes to ssnn/reports/memory.csv.
#
#   python -m ssnn.scripts.train_memory --seeds 7 19 43

MODEL_KEYS = ("oracle", "learned", "gated", "gated-unbalanced", "dense", "rnn", "gru", "gru-same-work", "gru-64")

parser = argparse.ArgumentParser(description="Experiment B of the SSNN: remembering registers across long, noisy gaps.")
parser.add_argument("--registers", type=positive_int, default=8, help="registers (default: 8)")
parser.add_argument("--values", type=positive_int, default=2, help="values a register can hold (default: 2, one bit)")
parser.add_argument("--noise", type=int, default=8, help="noise features on the NOOPs (default: 8)")
parser.add_argument("--experts", type=positive_int, default=16, help="experts of the learned SSNNs (default: 16)")
parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43], help="seeds of the data and models (default: 7 19 43)")
parser.add_argument("--epochs", type=positive_int, default=20, help="epochs of every model (default: 20)")
parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.003],
                    help="learning rates to train every model with, each from the start (default: 0.003)")
parser.add_argument("--max-distance", type=positive_int, default=30, help="largest distance of the training sequences (default: 30)")
parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS), help="models to train (default: all)")
parser.add_argument("--report", default=os.path.join(REPORTS, "memory.csv"), help="CSV file the results are added to")
args = parser.parse_args()

STATE_DIM, NUM_CLASSES = 32, args.values
task = MemoryTask(registers=args.registers, noise=args.noise, values=args.values)
env = environment()
ENVIRONMENT = {"commit": env["commit"], "numpy": env["numpy"], "python": env["python"], "machine": env["machine"],
               "blas_threads": env["blas_threads_env"]}


def ssnn(seed, **changes):
    config = make_config(seed=seed, input_dim=task.input_dim, state_dim=STATE_DIM, num_classes=NUM_CLASSES, **changes)
    return SequenceSSNN(SSNN(config)), config


# The models chosen with --models, in this order: (name, model, configuration of an SSNN or None)
def models(seed):
    oracle = ssnn(seed, num_experts=args.registers + 1, top_k=1, router="fixed", fixed_routes=task.oracle_routes())
    learned = ssnn(seed, num_experts=args.experts, top_k=2)
    gated = ssnn(seed, num_experts=args.experts, top_k=2, write="gated")
    gated_unbalanced = ssnn(seed, num_experts=args.experts, top_k=2, write="gated", balance_weight=0.0)
    dense = ssnn(seed, num_experts=args.experts, top_k=2, write="gated", router_gradient="dense")
    same_work = gru_hidden_for(learned[0].touched, task.input_dim, NUM_CLASSES)
    all_models = {
        "oracle": ("ssnn, oracle routing", *oracle), "learned": ("ssnn, learned router", *learned),
        "gated": ("ssnn, gated write", *gated), "gated-unbalanced": ("ssnn, gated, no balance", *gated_unbalanced),
        "dense": ("ssnn, gated, dense router gradient", *dense),
        "rnn": ("rnn (32)", TanhRNN(task.input_dim, STATE_DIM, NUM_CLASSES, seed=seed), None),
        "gru": ("gru (32)", GRU(task.input_dim, STATE_DIM, NUM_CLASSES, seed=seed), None),
        "gru-same-work": (f"gru ({same_work}), same work", GRU(task.input_dim, same_work, NUM_CLASSES, seed=seed), None),
        "gru-64": ("gru (64)", GRU(task.input_dim, 64, NUM_CLASSES, seed=seed), None),
    }
    return [all_models[key] for key in MODEL_KEYS if key in args.models]


results = []
print(f"{args.registers} registers of {args.values} values, {args.noise} noise features, distances up to "
      f"{args.max_distance} to train")

for seed in args.seeds:
    train, val, tests = experiment_memory(seed, task, max_distance=args.max_distance)
    print(f"\nSeed {seed}: test accuracy by distance from the last write")
    print(f"  {'model':34s} {'/event':>7}   " + "   ".join(f"{low:>3}-{high:<3}" for low, high in tests) + "     time")

    for learning_rate in args.learning_rates:
        for name, model, config in models(seed):  # new models for each learning rate
            label = name if len(args.learning_rates) == 1 else f"{name}, lr {learning_rate}"
            start = time.time()
            history = train_sequences(model, train, val, epochs=args.epochs, learning_rate=learning_rate, seed=seed,
                                      balance_weight=config["balance_weight"] if config else 0.0)
            seconds = time.time() - start
            accuracies = {distances: evaluate_sequences(model, examples)[1] for distances, examples in tests.items()}
            results.append((label, accuracies))

            # For an SSNN, how much the NOOPs of the gaps of 101 to 200 events touched and overwrote the experts the
            # QUERY reads
            leak = {"touched": "", "write": ""}
            if isinstance(model, SequenceSSNN):
                long = tests[(101, 200)]
                leak = memory_leak([model.forward(inputs)[1] for inputs, _, _ in long], [inputs for inputs, _, _ in long],
                                   [distance for _, _, distance in long], task.noop_index, config)

            print(f"  {label:34s} {model.touched:>7,}   " + "   ".join(f"{value * 100:6.1f}%" for value in accuracies.values())
                  + f"   {seconds:5.0f}s" + (f"   memory touched by {leak['touched'] * 100:.0f}% of the NOOPs of 101-200, "
                                                   f"mean write {leak['write']:.3f}" if leak["touched"] != "" else ""))
            append_row(args.report, {
                "date": datetime.now().isoformat(timespec="seconds"), "experiment": "B", "registers": args.registers,
                "values": args.values, "noise": args.noise, "seed": seed, "config": config_hash(config) if config else "",
                "model": name, "touched_per_event": model.touched, "epochs": args.epochs, "learning_rate": learning_rate,
                "max_distance": args.max_distance,
                "best_epoch": history["best_epoch"], "val_loss": min(history["val_loss"]),
                "val_accuracy": max(history["val_accuracy"]),
                **{f"accuracy_{low}_{high}": value for (low, high), value in accuracies.items()},
                "memory_leak_101_200": leak["touched"], "memory_write_101_200": leak["write"], "seconds": seconds,
                **ENVIRONMENT})

if len(args.seeds) > 1:
    print(f"\nMean over {len(args.seeds)} seeds:")
    for name in dict.fromkeys(name for name, _ in results):
        rows = [accuracies for model, accuracies in results if model == name]
        means = [np.mean([row[distances] for row in rows]) for distances in rows[0]]
        print(f"  {name:34s}          " + "   ".join(f"{value * 100:6.1f}%" for value in means))

print(f"\nResults added to {os.path.relpath(args.report)}")
