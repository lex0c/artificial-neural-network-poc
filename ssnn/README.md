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
python -m ssnn.scripts.train_capacity --seeds 7     # experiment D: about 15 minutes per seed
python -m ssnn.scripts.train_memory --seeds 7       # experiment B: about 20 minutes per seed
```

`train_static` adds one row per model and seed to `reports/experiments.csv`, with the configuration hash, the commit,
the versions and `OPENBLAS_NUM_THREADS`, so every result can be traced to the code and the setup that gave it. The
results below came from `OPENBLAS_NUM_THREADS=1`. `bench_cpu` does the same in `reports/benchmark.csv`, with the CPU
and the BLAS library and threads of each row, `train_capacity` in `reports/capacity.csv` and `train_memory` in
`reports/memory.csv`.

## Organization

| File | What it does |
| --- | --- |
| `config.py` | The default configuration, its checks and a hash to record with each result |
| `model.py` | Router with top-k, experts, integrator, decoder and the state of a session |
| `gradients.py` | The gradients of one event and over a sequence (through time), the balance loss of the router, L2 and gradient clipping |
| `recurrent.py` | A dense tanh RNN and a GRU with backpropagation through time, and the SSNN with the same interface |
| `tasks.py` | The data of experiments A, B and D |
| `training.py` | Training on independent examples or on sequences, keeping the epoch with the lowest validation loss |
| `metrics.py` | Accuracy, accuracy by regime and how the router uses the experts |
| `baselines.py` | Dense MLPs sized like the SSNN |
| `experiments.py` | Training and testing one model on a task, the same way for every model |
| `reporting.py` | The commit, the machine and the CSV rows recorded with every result |
| `inference.py` | Lean engines for inference, one event at a time: the selective SSNN, all its experts, and a dense RNN |
| `scripts/train_static.py` | Experiment A, with its baselines and ablations |
| `scripts/bench_cpu.py` | The latency benchmark of the engines |
| `scripts/train_capacity.py` | Experiment D: many regimes, each with its own rule |
| `scripts/train_memory.py` | Experiment B: remembering registers across long, noisy gaps |

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
- **Gradients of one event, or through time**: for independent examples (experiments A and D), the states before the
  event are treated as constants. For sequences (experiment B), backpropagation through time goes back over every
  event, and an expert that didn't run passes its gradient back unchanged, as it kept its state.
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

## Experiment D: many regimes, each with its own rule

The test of the one reason for the architecture: holding more capacity than a dense network of the same cost per
event can use. R regimes, each with a rule of its own given by a small random teacher network (8 tanh neurons over
the same 8 features), the regime as a one-hot vector, 400 examples per regime to train and 100 to test. The SSNN has
64 experts of 32 neurons, 2,048 in total, and runs 2 per event; a dense MLP with the 64 hidden neurons of 2 experts
has to share them between all the rules. Mean test accuracy over seeds 7, 19 and 43:

| Regimes | SSNN | SSNN, random router | MLP, same work (64 hidden) | MLP, same parameters per event | MLP, same total parameters |
| --- | --- | --- | --- | --- | --- |
| 4 | 86.7% | 69.8% | 91.7% | 93.5% | 93.3% |
| 16 | 87.6% | 57.5% | 92.5% | 93.8% | 93.2% |
| 64 | 84.6% | 54.3% | 92.2% | 93.0% | 92.6% |

What it shows:

- **The task didn't create the condition it was designed for.** The MLP with 64 hidden neurons didn't run out of
  capacity: with 64 regimes it is 0.4 points below the MLP with 2,960. And every dense network, however large, stays
  around 93%: the limit was the data, 400 examples per regime, not the capacity.
- **The SSNN is 5 to 8 points below every dense network**, for every number of regimes, and furthest with 64. The
  likely cause: each expert learns only from the examples the router sends it, while a dense network shares what the
  rules have in common. This cost of a mixture of experts dominates here.
- **The router learns**: about 30 points above the random router, with 0.9 to 1.7 bits of regime information
  against about 0.2 for the random router. With few events per pair of regime and expert, the estimate of that
  information is biased upwards, and the random router shows by how much.

A fair test of the capacity question would need many more examples per regime and harder teachers, until the MLP
with 64 hidden neurons reaches its limit while the larger ones keep improving: about 128,000 examples and 45 minutes
per seed with this training in Python. It hasn't been run.

## Experiment B: memory

Registers of one bit each. A sequence writes every register, then the one it will ask for once more, then has a gap
of NOOP events, with a rewrite of another register now and then, and ends with a QUERY of that register: the answer
is the last value written to it. The NOOPs carry 8 noise features that don't matter, which a recurrent network still
has to keep out of its memory. The models train on 2,000 sequences with distances (events from the last write to the
QUERY) up to 30, and are tested on distances up to 200, which they never saw.

- **SSNN, oracle**: one expert per register and one for the NOOPs, chosen by hand (`router: "fixed"`): what the
  architecture can do when the routing is right.
- **SSNN, learned**: 16 experts, 2 per event, the router learns which, trained by backpropagation through time.
- **SSNN, gated write**: the same, with each expert writing in proportion to the weight the router gives it
  (`write: "gated"`, `h = (1 - rho a) h + rho a n`), so a small weight leaves a state almost unchanged and the router
  gets a gradient through the write; with and without the balance loss.
- **RNN and GRUs**: a dense tanh RNN with 32 neurons (the work of one expert), and gated recurrent units with 32
  neurons, with as much work per event as the learned SSNN (28), and with 64.

Test accuracy with 8 registers, mean over seeds 7, 19 and 43:

| Model | Numbers touched per event | Distance 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | NOOPs touching the memory |
| --- | --- | --- | --- | --- | --- | --- |
| SSNN, oracle | 2,244 | 100% | 100% | 100% | 100% | 0% |
| SSNN, learned | 5,412 | 97.8% | 95.4% | 92.3% | 85.4% | 33% |
| SSNN, gated write | 5,412 | 99.2% | 97.6% | 92.9% | 83.9% | 29% |
| SSNN, gated write, no balance | 5,412 | 99.0% | 99.1% | 93.8% | 78.2% | 25% |
| RNN (32) | 2,244 | 87.1% | 85.4% | 86.1% | 80.4% | |
| GRU (28), same work | 5,332 | 93.8% | 93.0% | 95.4% | 93.6% | |
| GRU (32) | 6,468 | 93.9% | 92.4% | 95.0% | 92.9% | |
| GRU (64) | 19,012 | 98.1% | 97.6% | 98.4% | 98.1% | |

The last column is the share of the NOOPs of the longest gaps that selected one of the experts the final QUERY reads:
those events overwrite part of the memory the answer depends on.

What it shows:

- **With the right routing, the architecture keeps its promise**: the oracle remembers every register, at every
  distance, including those it never saw, with less work per event than any other model. Its memory experts don't
  run during the gaps, so nothing disturbs them, and their gradient reaches the write without fading (a test checks
  it is the same after 50 or 200 NOOPs).
- **With a learned router, the SSNN beats a dense network of the same work** within the distances it trained on: 2 to
  4 points above the GRU of the same work with the fixed write, 5 to 6 with the gated one, which gets close to the GRU
  with 3.5 times the work. It is the first experiment where it does.
- **But it doesn't keep that memory over longer gaps**, where the GRUs hold steady: 85% against 94% from 101 to 200
  events. The router hasn't learned to keep the memory experts away from the noise: about one NOOP in three selects
  one of them. The top-k choice itself gets no gradient, so nothing tells the router not to pick an expert. The gated
  write makes each such event less harmful, but doesn't prevent them, and over 200 events the small damages add up.
- With a single register and no noise, every model answers right at every distance, including the dense RNN, and even
  the learned SSNN, whose NOOPs go through the memory expert most of the time: holding one bit against an input that
  never changes is easy for any recurrent network. The noise and the number of registers are what make the task
  discriminate.

## Status of the hypotheses

| Hypothesis | Status |
| --- | --- |
| Running k of E experts reduces the computation per event | supported: the cost per event barely depends on E, 30 times less than running all 128 experts |
| That sparsity makes inference faster on a CPU, at a similar quality | not supported at this scale: 2 to 4 times slower than a dense RNN doing the same work per event in Python |
| The stored capacity buys quality | not supported in experiment D, where the SSNN stays below dense networks of the same work; the task turned out to be limited by data, so a larger version is still open |
| Persistent states help on tasks that need memory | supported with the right routing (perfect memory at every distance, with the least work), and in part with a learned router (above a GRU of the same work within the trained distances, below it beyond) |
| A learned routing specializes the experts | supported in experiments A and D (far above random routing, with regime information well above its bias), with partial specialization; in experiment B it doesn't learn to keep the memory experts out of the gaps |

## Conclusion so far

The mechanism works: the gradients are right, the selective execution matches the dense one, the router learns and
specializes the experts, and the stored capacity comes almost for free in latency.

Experiments A and D (independent examples) and the latency benchmark go against the SSNN: against a dense network
doing the same work per event, it loses on accuracy and on speed. Its stored capacity didn't buy quality, and the
specialization even cost accuracy in experiment D, as each expert learns from only part of the data.

Experiment B (memory) is where the architecture shows what it is for: experts that don't run keep their state exactly,
which gives a memory that neither fades nor gets disturbed, at a low cost per event. With the right routing it
remembers perfectly; with a learned router it beats a GRU of the same work on the distances it trained on, but leaks
over longer ones. The weak point isn't the architecture, it's teaching the router to protect its memory.

What stays open, in the order it would be worth testing:

1. **Teaching the router to protect its memory**: training on longer gaps, so the leaks cost more during training;
   then, if that isn't enough, gradients for the top-k choice itself (straight-through estimates or REINFORCE).
2. **Capacity at a larger scale**: many more examples per regime and harder rules.
3. **Speed** (compiled code, routing with locality, grouping the events of many sessions by expert): it matters only
   where the SSNN is better.

These are results of a first version: one laptop CPU, Python and NumPy, small tasks, hyperparameters not tuned, 3
seeds.
