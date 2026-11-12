import argparse
import os
import time
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from cli import positive_int
from ssnn.config import make_config
from ssnn.inference import AllExpertsEngine, DenseRNN, SelectiveEngine, rnn_hidden_for
from ssnn.model import SSNN, parameter_counts
from ssnn.reporting import REPORTS, append_row, environment


# Latency of inference, one event at a time (batch = 1), with random weights: it measures the cost of the
# architecture, not what it learned. For each configuration, four engines (ssnn/inference.py):
#   selective        the SSNN running its k selected experts
#   all experts      the same architecture running all E experts, in a few large operations
#   rnn, same work   a dense RNN that touches as many numbers per event as the selective SSNN
#   rnn, same total  a dense RNN with as many parameters in total as the SSNN (the same stored capacity)
# The configurations vary the number of experts, of selected experts and the size of the states, around d = 32,
# H = 256, E = 128, k = 2. Everything runs with one BLAS thread and with all of them, and each run feeds the same
# sequence of inputs to one session. After a warm-up, every event is timed; one row per configuration, engine and
# number of threads goes to ssnn/reports/benchmark.csv.
#
#   python -m ssnn.scripts.bench_cpu


parser = argparse.ArgumentParser(description="Latency of the SSNN on a CPU, against running all experts and dense RNNs.")
parser.add_argument("--events", type=positive_int, default=2000, help="timed events per repetition (default: 2000)")
parser.add_argument("--warmup", type=positive_int, default=500, help="events before timing (default: 500)")
parser.add_argument("--repetitions", type=positive_int, default=3, help="repetitions of the timed events (default: 3)")
parser.add_argument("--report", default=os.path.join(REPORTS, "benchmark.csv"), help="CSV file the results are added to")
args = parser.parse_args()

INPUT_DIM, NUM_CLASSES = 32, 2
CONFIGURATIONS = sorted({(E, 2, 256) for E in (16, 32, 64, 128)}      # more experts, the same work per event
                        | {(128, k, 256) for k in (1, 2, 4, 8)}       # more selected experts
                        | {(128, 2, H) for H in (32, 64, 128, 256)})  # larger states


def resident_memory_mb():
    try:
        with open("/proc/self/status") as file:
            return next(int(line.split()[1]) for line in file if line.startswith("VmRSS")) / 1024
    except (OSError, StopIteration):
        return float("nan")


def blas():
    info = threadpool_info()
    return {"blas": " ".join(f"{pool['internal_api']} {pool['version']}" for pool in info),
            "blas_threads": max((pool["num_threads"] for pool in info), default=0)}


# The latency of every event, in microseconds, after the warm-up
def measure(engine, state, inputs):
    for x in inputs[:args.warmup]:
        engine.step(x, state)

    latencies = np.empty(args.events * args.repetitions)
    timed = inputs[args.warmup:]
    for repetition in range(args.repetitions):
        for i, x in enumerate(timed):
            start = time.perf_counter_ns()
            engine.step(x, state)
            latencies[repetition * args.events + i] = time.perf_counter_ns() - start
    return latencies / 1000


def engines(E, k, H):
    config = make_config(input_dim=INPUT_DIM, state_dim=H, num_experts=E, top_k=k)
    model = SSNN(config)
    selective = SelectiveEngine(model)
    total = parameter_counts(config)["total"]
    same_work = DenseRNN(INPUT_DIM, rnn_hidden_for(selective.touched, INPUT_DIM, NUM_CLASSES), NUM_CLASSES)
    same_total = DenseRNN(INPUT_DIM, rnn_hidden_for(total, INPUT_DIM, NUM_CLASSES), NUM_CLASSES)
    return model, [("selective", selective, model.new_state()), ("all experts", AllExpertsEngine(model), model.new_state()),
                   ("rnn, same work", same_work, same_work.new_state()), ("rnn, same total", same_total, same_total.new_state())]


env = environment()
environment = {key: env[key] for key in ("commit", "cpu", "numpy", "python")}  # the columns of reports/benchmark.csv
inputs = np.random.default_rng(0).normal(0, 1, (args.warmup + args.events, INPUT_DIM)).astype(np.float32)
print(f"{environment['cpu']}, NumPy {environment['numpy']}; {args.events} events x {args.repetitions} after {args.warmup} of warm-up")

for threads in (1, None):
    with threadpool_limits(threads):
        blas_info = blas()
        print(f"\n{blas_info['blas']}, {blas_info['blas_threads']} BLAS thread(s); latency per event in microseconds, P50 (P95)")
        print(f"  {'E':>4} {'k':>2} {'H':>4} {'selective':>15} {'all experts':>17} {'rnn, same work':>17} {'rnn, same total':>17}"
              f"   selective vs all experts, vs rnn same work")

        for E, k, H in CONFIGURATIONS:
            model, runs = engines(E, k, H)
            p50 = {}
            cells = []
            for name, engine, state in runs:
                latencies = measure(engine, state, inputs)
                p50[name] = np.percentile(latencies, 50)
                cells.append(f"{p50[name]:8.1f} ({np.percentile(latencies, 95):6.1f})")
                append_row(args.report, {"date": datetime.now().isoformat(timespec="seconds"), "num_experts": E, "top_k": k, "state_dim": H,
                        "input_dim": INPUT_DIM, "engine": name, "touched_per_event": engine.touched,
                        "touched_bytes": engine.touched * 4, "p50_us": p50[name], "p95_us": np.percentile(latencies, 95),
                        "mean_us": latencies.mean(), "std_us": latencies.std(), "events_per_second": 1e6 / latencies.mean(),
                        "events": args.events, "repetitions": args.repetitions, "warmup": args.warmup,
                        "rss_mb": round(resident_memory_mb(), 1), **blas_info, **environment})
            faster = p50["all experts"] / p50["selective"]      # how many times faster than running all experts
            slower = p50["selective"] / p50["rnn, same work"]   # how many times the time of the dense RNN
            print(f"  {E:>4} {k:>2} {H:>4} {cells[0]:>15} {cells[1]:>17} {cells[2]:>17} {cells[3]:>17}   "
                  f"{faster:5.1f}x faster, {slower:4.2f}x the time of the rnn")

        # Where the time of one selective event goes, in the main configuration
        model, runs = engines(128, 2, 256)
        selective, state = runs[0][1], runs[0][2]
        parts = []
        for i, x in enumerate(inputs):
            marks = [time.perf_counter_ns()]
            selective.step(x, state, marks)
            if i >= args.warmup:
                parts.append(np.diff(marks))
        medians = np.median(parts, axis=0) / 1000
        print("  selective, E = 128, k = 2, H = 256, by part (median): " + ", ".join(
            f"{name} {value:.1f}" for name, value in zip(("router", "top-k", "experts", "integrator", "decoder"), medians)))

print(f"\nResults added to {os.path.relpath(args.report)}")
