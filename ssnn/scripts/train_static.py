import argparse
import os
from datetime import datetime

from cli import positive_int
from ssnn.config import config_hash, make_config
from ssnn.experiments import mlp_result, print_header, print_means, print_row, ssnn_result
from ssnn.model import parameter_counts
from ssnn.reporting import REPORTS, append_row, environment
from ssnn.tasks import experiment_a


# Experiment A: classification with 4 regimes (ssnn/tasks.py), for each seed:
#   ssnn          the SSNN, learned router, k = 2 of 16 experts
#   random router the same with k experts drawn at random per event (does the router learn something useful?)
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

parser = argparse.ArgumentParser(description="Experiment A of the SSNN: classification with 4 regimes.")
parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43], help="seeds of the data and models (default: 7 19 43)")
parser.add_argument("--epochs", type=positive_int, default=30, help="epochs of every model (default: 30)")
parser.add_argument("--examples", type=positive_int, nargs=3, default=[12000, 3000, 3000],
                    metavar=("TRAIN", "VAL", "TEST"), help="examples of each split (default: 12000 3000 3000)")
parser.add_argument("--report", default=os.path.join(REPORTS, "experiments.csv"), help="CSV file the results are added to")
args = parser.parse_args()

# The columns of reports/experiments.csv
RESULT_COLUMNS = ("model", "parameters", "per_event", "best_epoch", "val_loss", "test_loss", "test_accuracy",
                  "unused_experts", "usage_entropy", "regime_information", "seconds")
env = environment()
ENVIRONMENT = {"commit": env["commit"], "numpy": env["numpy"], "python": env["python"], "machine": env["machine"],
               "blas_threads": env["blas_threads_env"]}

all_results = []

for seed in args.seeds:
    data = experiment_a(seed, sizes=tuple(args.examples))
    base = make_config(seed=seed, epochs=args.epochs)
    print(f"\nSeed {seed}: {args.examples[0]:,} examples to train, {args.examples[1]:,} to validate, {args.examples[2]:,} to test")
    print_header(regimes=4)

    counts = parameter_counts(base)
    results = [
        ssnn_result("ssnn", base, data),
        ssnn_result("ssnn, random router", make_config(seed=seed, epochs=args.epochs, router="random"), data),
        ssnn_result("ssnn, all 16 experts", make_config(seed=seed, epochs=args.epochs, top_k=base["num_experts"]), data),
        mlp_result("mlp, same total", base, data, seed, target_parameters=counts["total"]),
        mlp_result("mlp, same per event", base, data, seed, target_parameters=counts["per_event"]),
        mlp_result("mlp, same hidden neurons", base, data, seed, hidden=base["top_k"] * base["state_dim"]),
    ]

    for result in results:
        print_row(result)
        append_row(args.report, {
            "date": datetime.now().isoformat(timespec="seconds"), "experiment": "A", "seed": seed,
            "config": config_hash(base), "epochs": args.epochs, "examples": "/".join(map(str, args.examples)),
            **{key: result[key] for key in RESULT_COLUMNS},
            **{f"regime_{regime}": value for regime, value in result["by_regime"].items()}, **ENVIRONMENT})
    all_results += results

if len(args.seeds) > 1:
    print(f"\nMean over {len(args.seeds)} seeds (test accuracy):")
    print_means(all_results)

print(f"\nResults added to {os.path.relpath(args.report)}")
