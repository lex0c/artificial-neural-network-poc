import argparse
import os
from datetime import datetime

import numpy as np

from cli import positive_int
from ssnn.config import config_hash, make_config
from ssnn.experiments import mlp_result, print_header, print_means, print_row, ssnn_result
from ssnn.model import parameter_counts
from ssnn.reporting import REPORTS, append_row, environment
from ssnn.tasks import experiment_capacity


# Experiment D: does the stored capacity of the SSNN buy quality? R regimes, each with a rule of its own given by a
# small random teacher network (ssnn/tasks.py), 400 examples per regime to train. For each number of regimes and seed:
#   ssnn             64 experts of 32 neurons (2,048 in total), 2 per event: it can give each regime experts of its own
#   random router    the same with 2 experts drawn at random per event: no regime gets experts of its own
#   mlp, same work   a dense MLP with the 64 hidden neurons the 2 experts compute per event: it has to share them
#                    between all the rules
#   mlp, per event   a dense MLP with as many parameters as the SSNN touches per event, router included
#   mlp, same total  a dense MLP with as many parameters as the SSNN has in total: the capacity, at a much higher cost
#                    per event
# If the architecture works, the SSNN should stay close to the last one and well above the first as the number of
# regimes grows. One row per model, number of regimes and seed goes to ssnn/reports/capacity.csv.
#
#   python -m ssnn.scripts.train_capacity --regimes 4 16 64 --seeds 7 19 43

parser = argparse.ArgumentParser(description="Experiment D of the SSNN: many regimes, each with its own rule.")
parser.add_argument("--regimes", type=positive_int, nargs="+", default=[4, 16, 64], help="numbers of regimes (default: 4 16 64)")
parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43], help="seeds of the data and models (default: 7 19 43)")
parser.add_argument("--epochs", type=positive_int, default=30, help="epochs of every model (default: 30)")
parser.add_argument("--per-regime", type=positive_int, nargs=3, default=[400, 100, 100], metavar=("TRAIN", "VAL", "TEST"),
                    help="examples per regime in each split (default: 400 100 100)")
parser.add_argument("--report", default=os.path.join(REPORTS, "capacity.csv"), help="CSV file the results are added to")
args = parser.parse_args()

NUM_FEATURES, NUM_EXPERTS, TOP_K, STATE_DIM = 8, 64, 2, 32
RESULT_COLUMNS = ("model", "parameters", "per_event", "best_epoch", "val_loss", "test_loss", "test_accuracy",
                  "worst_regime", "unused_experts", "usage_entropy", "regime_information", "seconds")
env = environment()
ENVIRONMENT = {"commit": env["commit"], "numpy": env["numpy"], "python": env["python"], "machine": env["machine"],
               "blas_threads": env["blas_threads_env"]}

results_by_regimes = {}

for num_regimes in args.regimes:
    for seed in args.seeds:
        data = experiment_capacity(seed, num_regimes, per_regime=tuple(args.per_regime))
        config = make_config(seed=seed, epochs=args.epochs, input_dim=NUM_FEATURES + num_regimes, num_experts=NUM_EXPERTS,
                             top_k=TOP_K, state_dim=STATE_DIM)
        counts = parameter_counts(config)
        print(f"\n{num_regimes} regimes, seed {seed}: {len(data[0][0]):,} examples to train "
              f"(regime information: at most {np.log2(num_regimes):.0f} bits)")
        print_header(regimes=num_regimes)

        results = [
            ssnn_result("ssnn", config, data),
            ssnn_result("ssnn, random router", {**config, "router": "random"}, data),
            mlp_result("mlp, same work", config, data, seed, hidden=TOP_K * STATE_DIM),
            mlp_result("mlp, same per event", config, data, seed, target_parameters=counts["per_event"]),
            mlp_result("mlp, same total", config, data, seed, target_parameters=counts["total"]),
        ]

        for result in results:
            print_row(result)
            append_row(args.report, {
                "date": datetime.now().isoformat(timespec="seconds"), "experiment": "D", "regimes": num_regimes,
                "seed": seed, "config": config_hash(config), "epochs": args.epochs,
                "per_regime": "/".join(map(str, args.per_regime)), **{key: result[key] for key in RESULT_COLUMNS},
                **ENVIRONMENT})
        results_by_regimes.setdefault(num_regimes, []).extend(results)

for num_regimes, results in results_by_regimes.items():
    if len(args.seeds) > 1:
        print(f"\n{num_regimes} regimes, mean over {len(args.seeds)} seeds (test accuracy):")
        print_means(results)

print(f"\nResults added to {os.path.relpath(args.report)}")
