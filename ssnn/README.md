# SSNN: Sparse Stateful Neural Network

A proof of concept of a modular recurrent architecture: for each event, a router picks a few recurrent modules
(experts), only those update their state, and a global state integrates their outputs. The question it tests: **can
selective processing with persistent state learn useful tasks and speed up inference on a CPU, in a fair comparison
with dense alternatives?** A negative answer is a valid result.

Python and NumPy only, with the gradients written by hand and checked numerically.

## The model

For each event `x_t`, with `E` experts of state size `H`:

```
router      q = concat(x_t, g_(t-1));  z = W_r q + b_r;  S = the k largest z;  a = softmax(z[S] / tau)
experts     for i in S:  n_i = tanh(W_x[i] x_t + W_h[i] h_i + b[i]);  h_i = (1 - rho) h_i + rho n_i
            the other experts keep their state unchanged
integrator  v = sum over S of a_i h_i;  g_t = (1 - beta) g_(t-1) + beta v
decoder     p = softmax(W_y concat(x_t, g_t) + b_y)
```

Only `k` of the `E` experts run, so an event touches about `k / E` of the expert weights, while every expert keeps its
own state between the events of a session. The router is trained through the softmax of the selected experts, plus a
balance loss on the softmax of all of them, which keeps it from always picking the same few.

## Running

From the root of the repository:

```sh
python -m unittest discover -s ssnn/tests -t .      # the tests of the SSNN (python -m unittest runs them too)
python -m ssnn.scripts.train_static --seeds 7 19 43 # experiment A: about 6 minutes per seed
python -m ssnn.scripts.bench_cpu                    # the latency benchmark: about 6 minutes
```

`train_static` adds one row per model and seed to `reports/experiments.csv`, with the configuration hash, the commit,
the versions and `OPENBLAS_NUM_THREADS`, so every result can be traced to the code and the setup that gave it. The
results below came from `OPENBLAS_NUM_THREADS=1`. `bench_cpu` does the same in `reports/benchmark.csv`, with the CPU
and the BLAS library and threads of each row.

## Organization

| File | What it does |
| --- | --- |
| `config.py` | The default configuration, its checks and a hash to record with each result |
| `model.py` | Router with top-k, experts, integrator, decoder and the state of a session |
| `gradients.py` | The gradients of one event, the balance loss of the router, L2 and gradient clipping |
| `tasks.py` | The data of experiment A |
| `training.py` | Training on independent examples, keeping the epoch with the lowest validation loss |
| `metrics.py` | Accuracy, accuracy by regime and how the router uses the experts |
| `baselines.py` | Dense MLPs sized like the SSNN |
| `inference.py` | Lean engines for inference, one event at a time: the selective SSNN, all its experts, and a dense RNN |
| `scripts/train_static.py` | Experiment A, with its baselines and ablations |
| `scripts/bench_cpu.py` | The latency benchmark of the engines |

It shares `activations.py` (stable softmax), `optimizers.py` (Adam), `progress.py` and `gradcheck.py` with the rest of
the repository. Those modules keep float32 values in float32 and don't limit the BLAS threads; only `baselines.py`
imports `feedforward.py`, which does.

## Design choices

- **A package at the root of the repository**, so it can share the modules above. Its files have other names than the
  shared ones, so a `data.py` or a `losses.py` always means the shared one.
- **Reproducible top-k**: the largest logits, ties broken by the lower index (`np.argpartition` doesn't keep a stable
  order).
- **Copies of the state**: `step(x, state)` returns a new state and leaves the given one unchanged, which rules out a
  class of bugs where one session changes another.
- **Gradients of one event**: the states before the event are treated as constants. That is enough for tasks whose
  answer depends on the current event (experiment A), not for those that need to remember, which need
  backpropagation through time.
- **Fair comparisons**: every model, SSNN and baselines alike, trains on the same data with the same epochs,
  minibatches and learning rate, and ends with the parameters of its epoch with the lowest validation loss; the test
  set is used once, at the end.
- **Dense baselines of quality**: the MLPs use the `FeedForward` of this repository (float64, one BLAS thread). They
  are fine for comparing accuracy, not latency: a latency benchmark needs baselines written in the same lean style as
  the SSNN.
- **Gradient checks in float64**: central differences in float32 are too coarse to check a gradient.
- **One event at a time in training too**: the minibatch is a Python loop over its examples, which follows the
  equations line by line. Forward and gradients of one event take about 150 microseconds with the default
  configuration, so experiment A trains in about a minute per model. A vectorized minibatch would be a second
  implementation to keep equivalent; it is worth it only if training on sequences needs it.

## Experiment A: classification with 4 regimes

Each example has four numbers `u1..u4` and a regime from 0 to 3, given as a one-hot vector, and the binary label
follows a different rule in each regime (`u1 * u2 > 0`, `u1 + u2 > 0`, `u3 * u4 > 0`, `|u1| > |u2|`). The regime is in
the input on purpose, so the routing can be tested; the task can't show experts appearing on their own. Every example
starts from the zero state, so the recurrent weights play no part. 12,000 examples to train, 3,000 to validate and
3,000 to test, 30 epochs, seeds 7, 19 and 43:

| Model | Parameters | Used per event | Test accuracy | Regime information |
| --- | --- | --- | --- | --- |
| SSNN, learned router, 2 of 16 experts | 21,730 | 3,362 | 96.4% (95.6 to 96.9) | 1.08 bits |
| SSNN, random router | 21,730 | 3,362 | 92.7% (91.7 to 93.4) | 0.00 bits |
| SSNN, all 16 experts | 21,730 | 21,730 | 98.3% (98.0 to 98.5) | 0.00 bits |
| MLP, same total parameters (1,975 hidden) | 21,727 | 21,727 | 97.7% (97.5 to 98.1) | |
| MLP, same parameters per event (305 hidden) | 3,357 | 3,357 | 98.1% (97.9 to 98.3) | |
| MLP, same hidden neurons per event (64) | 706 | 706 | 97.3% (96.7 to 97.6) | |

The regime information is the mutual information between the regime of an event and the experts it selects: 0 when
every regime uses the experts in the same proportions, 2 bits when each regime has experts of its own. The 3,362
parameters per event of the SSNN include the recurrent weights of its 2 experts, which do nothing here: it computes
with 64 hidden neurons per event, hence the last baseline.

What it shows:

- **The router learns something useful**: on every seed the learned routing beats the random one, by 3.7 points on
  average, and its choice of experts carries about 1.1 of the 2 bits about the regime. The experts specialize, but
  only in part: regimes still share experts.
- **The selection costs accuracy**: with 2 of its 16 experts the SSNN is 1.9 points below itself with all 16, and below
  every dense baseline, including the MLP with the same 64 hidden neurons per event and 30 times fewer parameters
  (0.9 points). On this task, a small dense network does the same job better.
- The regime with the most errors is the product `u3 * u4 > 0` (93.9% to 95.4%), a rule a single tanh layer has to
  bend around.

Training times say nothing about inference speed, which needs its own benchmark.

## Latency on a CPU

Inference of one event at a time, with random weights (the cost of the architecture, not what it learned), d = 32
inputs, on an Intel i7-9750H (12 MB of L3 cache) with one BLAS thread. Four engines (`inference.py`): the selective
SSNN, the same architecture running all its experts, a dense RNN that touches as many numbers per event as the
selective SSNN (the same work), and a dense RNN with as many parameters as the SSNN in total (the same stored
capacity). Median latency per event, over 6,000 events after a warm-up:

| Experts (E), selected (k), state (H) | Selective | All experts | Dense RNN, same work | Dense RNN, same total |
| --- | --- | --- | --- | --- |
| E = 16, k = 2, H = 256 | 42 µs | 116 µs | 20 µs | 79 µs |
| E = 32, k = 2, H = 256 | 44 µs | 274 µs | 20 µs | 232 µs |
| E = 64, k = 2, H = 256 | 51 µs | 762 µs | 21 µs | 476 µs |
| E = 128, k = 2, H = 256 | 61 µs | 1,826 µs | 23 µs | 1,216 µs |
| E = 128, k = 1, H = 256 | 39 µs | 1,830 µs | 18 µs | 1,217 µs |
| E = 128, k = 8, H = 256 | 178 µs | 1,833 µs | 47 µs | 1,221 µs |
| E = 128, k = 2, H = 32 | 31 µs | 61 µs | 9 µs | 29 µs |

`reports/benchmark.csv` has all 10 configurations, with P95, and the same runs with 12 BLAS threads, which only help
the large dense RNN (1,216 to 1,006 µs) and slow down running all the experts.

What it shows:

- **The selection does what it promises against the full architecture**: the latency barely grows with the number of
  experts (42 to 61 µs from 16 to 128 experts), while running all of them grows with E. With 128 experts the
  selective SSNN is 30 times faster than running all of them, and 20 times faster than a dense RNN with the same
  stored capacity (37 MB of weights).
- **Against a dense RNN doing the same work per event, it is 2 to 4 times slower** in every configuration. Where the
  61 µs of the main configuration go: router 6, top-k 8, the 2 experts 39, integrator 2, decoder 7.
- **Two causes**: an event makes about 30 small NumPy calls, against a few for the RNN, and the overhead of each one
  dominates when the products are small (with H = 32, 31 against 9 µs). And the experts the router picks change from
  event to event, so with 37 MB of expert weights, more than the cache holds, theirs come from the main memory: in a
  separate measurement, forcing the router to always pick the same 2 experts brought their time from about 55 to
  30 µs. The first cause belongs to a Python implementation; the second to the architecture, unless the routing
  reuses the same experts.

## Status of the hypotheses

| Hypothesis | Status |
| --- | --- |
| Running k of E experts reduces the computation per event | supported: the cost per event barely depends on E, 30 times less than running all 128 experts |
| That sparsity makes inference faster on a CPU, at a similar quality | not supported at this scale: 2 to 4 times slower than a dense RNN doing the same work per event, and less accurate than dense networks of the same work in experiment A |
| Persistent states help on tasks that need memory | not tested yet: needs a memory task and backpropagation through time |
| A learned routing specializes the experts | supported in experiment A (better than random routing, 1.1 bits of regime information), with partial specialization |

## Conclusion so far

The mechanism works: the gradients are right, the selective execution matches the dense one, the router learns and
specializes the experts, and the stored capacity comes almost for free in latency. But against a dense network
doing the same work per event, the SSNN loses on both fronts, speed and accuracy, even to an MLP of 706 parameters.

The one reason for the architecture is to hold much more capacity than a dense network of the same cost per event can
use. In experiment A that capacity bought nothing, because the task is too simple. So the question that decides is
still open: is there a task where the extra capacity buys quality? Next, in this order:

1. **A task that needs capacity**: dozens of regimes, each with its own rule, so that a dense network of the same cost
   per event can't learn them all and the SSNN can give each regime experts of its own. If the SSNN doesn't win
   there either, the efficiency hypothesis can be closed.
2. **A memory task**, to test the persistent states, which experiment A doesn't use.
3. **Only then, speed** (compiled code, routing with locality): being faster matters only where the SSNN is better.

These are results of a first version: one laptop CPU, Python and NumPy, small tasks, hyperparameters not tuned, 3
seeds.

## Notes for a memory task

With `beta = 0.6`, the global state keeps 0.4 of its past at every event: after 10 events, what an event wrote weighs
0.4^10, about 0.0001. A memory across many events then has to live in an expert that the router *doesn't* select in
the meantime and selects again when the answer is asked, which is exactly what the architecture promises, but hard to
learn when only the selected experts get gradients. A smaller `beta` is worth trying there.
