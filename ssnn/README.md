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
balance loss on the softmax of all of them, which keeps it from always picking the same few. With
`router_gradient: "dense"` it also learns which experts to choose: for each expert left out of an event, an estimate
of what adding it to the selection would do to the loss.

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
| `gradients.py` | The gradients of one event and over a sequence (through time), the dense router gradient, the balance loss of the router, L2 and gradient clipping |
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
- **A gradient for the choice of the experts** (`router_gradient: "dense"`): the top-k choice isn't differentiable, so
  the softmax of the selected experts only teaches the router how to share the weight among them. For each expert e
  left out, the dense gradient takes the first-order change of the loss if e were added to the selection with a small
  weight, through what e would read and write (with the gated write) and the gradient its state gets from the
  following events, and pushes the logit of e up if that would lower the loss. It is an estimate: only the router
  gets it, and the gradients of every other parameter stay exact. It costs computing what every expert would write, at
  every event of the training (40 to 50% more training time with 32 experts), and nothing at inference. In experiment
  B, it helped the router when each write is an input feature of its own, and did worse when the input names the
  register.
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

The GRUs of the first three comparisons below start with the bias of their update gate at 0, so they start by
forgetting half of their state at every event. With the usual initialization, which makes them keep it, they beat the
learned SSNN on both memory tasks: see [A GRU that keeps its state by default](#a-gru-that-keeps-its-state-by-default).

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

### Training on longer gaps

If the router leaks because the training never made a leak costly over a long gap, training on longer gaps should
help. The same experiment with training distances up to 100 instead of 30 (`--max-distance 100`), for the learned
router, the gated write and the GRUs, seeds 7, 19 and 43:

| Model | Distance 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | NOOPs touching the memory | Their mean write on it |
| --- | --- | --- | --- | --- | --- | --- |
| SSNN, gated write | 98.6% | 98.8% | 98.1% | 97.1% (96.7 to 97.7) | 20% | 0.035 |
| SSNN, learned, fixed write | 87.4% | 87.2% | 87.9% | 84.7% | 29% | 0.102 |
| GRU (28), same work | 93.6% | 92.6% | 95.4% | 92.9% (91.7 to 94.7) | | |
| GRU (64) | 77.8% | 74.3% | 78.4% | 77.1% | | |

- **With the gated write, the SSNN now beats the GRU of the same work at every distance** (a GRU that starts by
  forgetting, see the note above), including those it never saw: 97.1% against 92.9% from 101 to 200 events, and its
  worst seed is above the best seed of the GRU.
- **The router learned to protect its memory, but not by avoiding it**: the NOOPs still select a memory expert 20% of
  the time, but with a small weight, so their mean write on it fell to 0.035, against 0.102 with the fixed write (whose
  write is always rho = 0.35 when selected). Protecting the memory needed both: the gated write, which gives the router
  a gradient for how much an expert writes, and long gaps in the training, which make a careless write costly.
- **With the fixed write, longer gaps don't help**, and the training got worse: with the top-k alone, nothing tells
  the router how to protect a memory.
- **The GRU with 64 neurons didn't train well on the longer sequences** at this learning rate (0.003, the same for
  every model): on one seed its best epoch was the first, and it ended near chance; on the others it stopped improving
  around epoch 10. Trained on gaps up to 30 it had reached 98% at every distance, so this says nothing about its
  capacity. The learning rate wasn't the cause (next section); its initialization was (see the note above).

### Fair learning rates, longer tests and a harder task

To see whether that holds, the same comparison with each model trained at learning rates 0.001 and 0.003 and keeping,
on each seed, the one with the lowest validation loss; tested on distances up to 500; and on a harder task, 16
registers holding one of 4 values each (a guess is right 25% of the time), with 32 experts for the learned SSNN. Seeds
7, 19 and 43, mean test accuracy (worst seed):

8 registers of 2 values:

| Model | Numbers touched per event | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 |
| --- | --- | --- | --- | --- | --- | --- |
| SSNN, gated write | 5,412 | 98.6% | 98.8% | 98.1% | 97.1% | 93.7% (91.0) |
| GRU (28), same work | 5,332 | 93.6% | 92.6% | 95.4% | 92.9% | 92.8% (91.3) |
| GRU (64) | 19,012 | 83.8% | 85.9% | 89.1% | 85.8% | 85.1% |

16 registers of 4 values:

| Model | Numbers touched per event | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 |
| --- | --- | --- | --- | --- | --- | --- |
| SSNN, oracle | 4,392 | 100% | 100% | 100% | 100% | 100% |
| SSNN, gated write | 12,200 | 71.6% | 71.3% | 68.9% | 59.8% | 54.9% |
| GRU (32), same work | 12,200 | 25.7% | 23.2% | 23.6% | 25.2% | 22.9% |
| GRU (64) | 30,184 | 25.4% | 26.1% | 24.7% | 25.4% | 25.1% |

- **The learning rate didn't change the comparison**: the gated SSNN and the GRU of the same work kept 0.003 on every
  seed. With 8 registers the SSNN stays ahead up to 200 events, and ties with the GRU from 201 to 500 (93.7% against
  92.8%, the worst seeds 91.0% and 91.3%): over gaps 5 times longer than any in its training, its memory starts to
  fade too.
- **The GRU with 64 neurons stays unstable on these long sequences** at both learning rates, below the GRU of the same
  work; trained on gaps up to 30 it had reached 98%.
- **On the harder task, neither GRU got off the ground**: their validation loss stayed at ln 4 = 1.386, a constant
  answer, through the 20 epochs. The SSNN learned part of it, about 70% on the trained distances, and the oracle all
  of it, at every distance. The GRUs' failure came from their initialization, and the SSNN's gap from its learned
  routing (next sections).

These rows were run on the working tree of commit 48c4158 before it was committed (the code didn't change in between),
so `reports/memory.csv` gives that commit for them.

### Why the learned routing falls short

A look at the routing of one trained gated SSNN (16 registers of 4 values, seed 7, learning rate 0.003; a diagnosis
outside the reports): for 28% of the test sequences of 1 to 10 events, the QUERY selects none of the experts the last
write of its register selected, and answers right only 54% of the time, against 80% when it does. The writes of the
4 values of a register mostly go to different pairs of experts, so the value is in which expert was written as much
as in its state, and the QUERY reads only some of them. A few experts serve many registers, which overwrite each other.

That is what the router can't fix with the gradient of the selected experts alone: it learns how to share the weight
between the 2 experts it picked, but nothing tells the QUERY of a register that another expert holds what it needs,
or a write that the QUERY will read another expert.

### A gradient for the choice of the experts

The same comparison with `router_gradient: "dense"` (see the design choices): the gated SSNN, with an estimate for
each expert left out of an event of what adding it would do to the loss. Same data, 20 epochs, learning rates 0.001
and 0.003 kept by the validation loss of each seed (0.003 on every seed), mean test accuracy (worst seed):

| Model | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 | Validation loss |
| --- | --- | --- | --- | --- | --- | --- |
| 16 registers of 4 values, dense router gradient | 81.1% (79.7) | 77.8% (73.0) | 80.6% (76.7) | 76.4% (75.3) | 66.2% (62.7) | 0.498 |
| 16 registers of 4 values, selected experts only | 71.6% (66.3) | 71.3% (67.0) | 68.9% (65.0) | 59.8% (50.0) | 54.9% (47.0) | 0.767 |
| 8 registers of 2 values, dense router gradient | 98.2% (97.0) | 97.9% (97.3) | 98.7% (98.3) | 98.3% (97.7) | 97.4% (96.3) | 0.044 |
| 8 registers of 2 values, selected experts only | 98.6% (98.0) | 98.8% (98.3) | 98.1% (96.7) | 97.1% (96.7) | 93.7% (91.0) | 0.043 |

- **On the harder task it gains 6 to 17 points at every distance**, and its worst seed is above the mean of the
  selected-only gradient everywhere. In the diagnosis above, the QUERYs that read none of the experts of the last write
  fell from 28% to 18%.
- **On the easier one it changes less than a point within the trained distances, and holds the memory better beyond
  them**: 97.4% against 93.7% from 201 to 500 events.
- **It doesn't come from fewer leaks**: the NOOPs of the gaps of 101 to 200 events touch the memory more often (16%
  against 12% with 16 registers, 31% against 20% with 8), with a mean write on it of 0.028 and 0.055 (against 0.021
  and 0.035). What improved is that the QUERY finds the expert of the write.
- **It still trains slowly**: its best epoch was the 16th to the 20th of 20 on every seed, and it stays far from the
  oracle (100% at every distance) on the harder task.

### A GRU that keeps its state by default

The GRUs above start with the bias of the update gate at 0, so z starts around 0.5: each event keeps half of the state,
and the gradient of an old state halves at every event. Trained 60 epochs instead of 20, those GRUs still didn't leave
chance on the harder task (their best epoch was again the 1st to the 4th); in a run that printed it, the training loss
fell while the validation loss rose, so they memorized the training sequences instead of learning to remember. The
usual initialization for long memories starts that bias positive. With 2 (z around 0.88, `--gru-update-bias 2`), the
same protocol (20 epochs, learning rate kept by validation, 0.003 on every seed):

| Model | Numbers touched per event | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 |
| --- | --- | --- | --- | --- | --- | --- |
| **8 registers of 2 values** | | | | | | |
| GRU (28), same work, update bias 2 | 5,332 | 99.8% (99.3) | 99.9% (99.7) | 100% | 100% | 99.7% (99.3) |
| GRU (64), update bias 2 | 19,012 | 100% | 100% | 100% | 100% | 99.9% (99.7) |
| SSNN, gated, dense router gradient | 5,412 | 98.2% (97.0) | 97.9% (97.3) | 98.7% (98.3) | 98.3% (97.7) | 97.4% (96.3) |
| **16 registers of 4 values** | | | | | | |
| SSNN, oracle | 4,392 | 100% | 100% | 100% | 100% | 100% |
| GRU (32), same work, update bias 2 | 12,200 | 86.4% (83.3) | 86.7% (86.3) | 88.8% (87.0) | 87.9% (84.3) | 87.6% (79.7) |
| GRU (64), update bias 2 | 30,184 | 87.7% (86.0) | 88.3% (85.7) | 91.1% (89.0) | 89.6% (87.7) | 92.1% (89.0) |
| SSNN, gated, dense router gradient | 12,200 | 81.1% (79.7) | 77.8% (73.0) | 80.6% (76.7) | 76.4% (75.3) | 66.2% (62.7) |

- **With it, the GRU of the same work beats the learned SSNN on both tasks, at every distance**: by 1.3 to 2.3 points
  with 8 registers, where it is nearly perfect, and by 5 to 21 points with 16 registers of 4 values. The comparisons
  above where the learned SSNN came ahead were against GRUs that start by forgetting.
- **The GRU with 64 neurons trains well too**, at 0.003: its instability on long sequences came from the same
  initialization.
- **The oracle stays above every model**: 100% at every distance on the harder task with about a third of the work of
  the GRU of the same work, where the GRUs reach 86 to 92%. The architecture can hold such a memory; a learned router
  doesn't find the routing that does it.

The rows of the last two sections were also run before their code was committed, on working trees identical to commit
e9b567a (the dense router gradient, and the GRUs trained 60 epochs) and 5d14da9 (the GRUs with the update bias), so
`reports/memory.csv` gives those commits for them.

### An input that names the register

In this task, each pair of register and value, and each QUERY, is an input feature of its own: nothing is common to
the writes and the QUERY of a register, and the router has to align 5 unrelated columns of its weights. In a real
memory, the key would be a feature of its own. The same comparison on 16 registers of 4 values with `--encoding keyed`:
the register is a feature shared by its writes and its QUERY, and the value and the operation are others (31 inputs
instead of 89, so every model is smaller). GRUs with an update bias of 2, the same protocol, mean test accuracy (worst
seed):

| Model | Numbers touched per event | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 | Validation loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SSNN, oracle | 2,304 | 100% | 100% | 100% | 100% | 100% | 0.001 |
| SSNN, gated write | 6,400 | 96.8% (92.3) | 95.8% (91.7) | 92.7% (87.0) | 89.4% (85.3) | 80.9% (70.3) | 0.129 |
| SSNN, gated, dense router gradient | 6,400 | 88.4% (81.3) | 87.1% (76.7) | 85.1% (74.7) | 83.1% (72.3) | 71.6% (53.3) | 0.319 |
| GRU (32), same work | 6,400 | 91.1% (89.7) | 89.8% (88.0) | 91.7% (87.3) | 89.1% (88.3) | 90.4% (85.7) | 0.318 |
| GRU (64) | 18,816 | 92.9% (87.7) | 93.2% (87.7) | 95.1% (91.7) | 93.1% (89.3) | 85.7% (81.0) | 0.214 |

- **The encoding was most of the gap**: the gated SSNN went from 71.6% to 96.8% from 1 to 10 events, and from 54.9% to
  80.9% from 201 to 500, while the GRU of the same work gained 5 points or less.
- **Now the learned SSNN beats the GRU of the same work up to 100 events**, by 1 to 6 points, ties with it from 101 to
  200 (89.4% against 89.1%), and has the lowest validation loss of all the learned models. Beyond the trained
  distances its memory fades faster: 80.9% against 90.4% from 201 to 500, with the NOOPs of the long gaps touching
  the memory 19% of the time.
- **Its seeds are bimodal**: a seed either finds a routing close to the oracle's, with a validation loss around 0.02
  and 94 to 95% even from 201 to 500 events, or falls well short of it. One seed in three found it with each router
  gradient (seed 19 with the selected-only one, seed 7 with the dense one), so the difference between their means is
  mostly which seeds found it: 3 seeds don't tell which gradient is better here (8 do, in the next section). One GRU
  with 64 neurons got close too (seed 43, validation loss 0.024, above 99% up to 200 events).
- **The SSNNs were still improving at the 20th epoch** (best epochs 16 to 20), while the GRUs of the same work peaked
  between the 6th and the 9th at 0.003 and then overfit.

These rows were run on the working tree of commit a0c67c4 before it was committed (the code didn't change in between),
so `reports/memory.csv` gives that commit for them.

### More seeds and longer trainings

Three seeds left open how often the learned router finds the oracle's routing, whether more epochs get the other
seeds there, and which router gradient is better. The same keyed task with 8 seeds (1 to 5, 7, 19 and 43) and 40
epochs, at a learning rate of 0.003, which had the lowest validation loss on every seed of the last section for these
three models. Mean test accuracy (worst seed):

| Model | Numbers touched per event | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 | Validation loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SSNN, gated write | 6,400 | 97.9% (94.0) | 97.5% (95.0) | 96.8% (91.7) | 93.7% (86.7) | 80.7% (65.7) | 0.071 |
| SSNN, gated, dense router gradient | 6,400 | 90.8% (82.3) | 91.1% (80.7) | 90.8% (82.7) | 88.9% (79.7) | 78.2% (63.0) | 0.225 |
| GRU (32), same work, update bias 2 | 6,400 | 91.1% (89.7) | 91.2% (88.0) | 91.5% (87.3) | 90.4% (88.3) | 89.6% (85.7) | 0.318 |

- **The learned SSNN beats the GRU of the same work on every seed up to 100 events**, by 5 to 7 points on average, with
  a validation loss 4 times lower (lower on all 8 seeds), and on 5 seeds of 8 from 101 to 200 (one more is a tie).
  Beyond 200 events, the GRU wins on 7 seeds of 8: 89.6% against 80.7%.
- **The selected-only router gradient is the better one here**: a lower validation loss than with the dense one on 6
  seeds of 8, in two thirds of the time (2,082 s against 3,136 s per run, 12 runs at a time). The dense gradient
  helped when the router had to align unrelated input features, not when the input already names the register.
- **More epochs helped the SSNN, not the GRU**: on the 3 seeds of the last section, the validation loss of the gated
  SSNN halved (0.129 to 0.063) and its accuracy from 201 to 500 went from 80.9% to 85.2%; 6 of its 8 seeds had their
  best epoch at the 37th or later, so it was still improving. The GRU peaked between the 6th and the 13th epoch, so
  its rows are the same as with 20.
- **Finding the oracle's routing is less all-or-nothing than 3 seeds suggested**: 3 seeds of 8 reached a validation
  loss below 0.05 with the gated SSNN (2 of 8 with the dense gradient), at 99.9 to 100% up to 100 events and 98% from
  101 to 200; the other 5 still averaged 95 to 97% up to 100 events. Even those 3 fade beyond 200 events (86.7%, and
  67.7% on one of them). Over all 8 seeds, the NOOPs of the gaps of 101 to 200 touched the memory 17% of the time,
  with a mean write of 0.032, but that doesn't explain the fading: the two seeds that fade the most (65.7% and 67.7%
  from 201 to 500) are among those with the fewest (2% and 7%, against up to 47%).

## Status of the hypotheses

| Hypothesis | Status |
| --- | --- |
| Running k of E experts reduces the computation per event | supported: the cost per event barely depends on E, 30 times less than running all 128 experts |
| That sparsity makes inference faster on a CPU, at a similar quality | not supported at this scale: 2 to 4 times slower than a dense RNN doing the same work per event in Python |
| The stored capacity buys quality | not supported in experiment D, where the SSNN stays below dense networks of the same work; the task turned out to be limited by data, so a larger version is still open |
| Persistent states help on tasks that need memory | supported with the right routing: chosen by hand, perfect memory at every distance (up to 500 events) with the least work, above GRUs with 7 times the work. With a learned router, it depends on the input: when the input names the register, the SSNN beats a GRU of the same work up to 100 events on every seed of 8, with a validation loss 4 times lower, but not beyond 200; when each write is a feature of its own, a GRU of the same work initialized to keep its state beats it at every distance |
| A learned routing specializes the experts | supported in experiments A and D (far above random routing, with regime information well above its bias), with partial specialization; in experiment B it learns to protect its memory with the gated write, by writing little to it rather than avoiding it, but, with one input feature per write, not to give each register experts that its writes and its QUERY share (the dense router gradient helps, by 6 to 17 points, without closing the gap). With an input that names the register, 3 seeds of 8 find a routing close to the oracle's and the others come near it, without the dense router gradient, which does worse there |

## Conclusion so far

The mechanism works: the gradients are right, the selective execution matches the dense one, the router learns and
specializes the experts, and the stored capacity comes almost for free in latency.

Experiments A and D (independent examples) and the latency benchmark go against the SSNN: against a dense network
doing the same work per event, it loses on accuracy and on speed. Its stored capacity didn't buy quality, and the
specialization even cost accuracy in experiment D, as each expert learns from only part of the data.

Experiment B (memory) shows what the architecture is for, and what it lacks. Experts that don't run keep their state
exactly: with the routing chosen by hand, the SSNN remembers perfectly, at every distance and on every task tried,
with less work per event than any other model, and above GRUs with 7 times its work on the harder task. Whether a
learned router finds that routing depends on the input:

- When each write of a register and a value is an input feature of its own, it doesn't. A gradient for the choice of
  the experts (the dense router gradient) took the learned SSNN from about 70% to about 80% on the harder task, but a
  GRU of the same work, with the usual initialization that makes it keep its state, beats it at every distance:
  99.7 to 100% against 97 to 99% with 8 registers, 86 to 89% against 66 to 81% with 16 registers of 4 values. The
  earlier comparisons where the learned SSNN came ahead were against GRUs that start by forgetting.
- When the input names the register, as the key of a memory would, the learned SSNN beats the GRU of the same work up
  to 100 events on every seed of 8 (97 to 98% against about 91%, after 40 epochs), with a validation loss 4 times
  lower, and 3 seeds of 8 find a routing close to the oracle's. There, the selected-only router gradient does
  better than the dense one. Beyond the distances it trained on, its memory fades faster than the GRU's.

So the SSNN can be the better memory, at the same work per event, when its input says what to remember and within the
distances it trained on; a well-initialized GRU stays the safer choice otherwise. What stays open, in the order it
would be worth testing:

1. **Why the learned memory fades beyond the trained distances**, now the main gap: the NOOPs of the long gaps touch
   the memory experts about one time in six, with a small write that adds up, but the seeds that fade the most are
   among those they touch the least. The rewrites of other registers, more frequent the longer the gap, may touch
   the experts of the queried register too when registers share experts; the leak metric counts only the NOOPs.
2. **Longer trainings**: the gated SSNN was still improving at the 40th epoch.
3. **Capacity at a larger scale**, and **speed** (compiled code, routing with locality, grouping the events of many
   sessions by expert), which matters only where the SSNN is better: in Python, an SSNN event still takes 2 to 4
   times as long as a dense network doing the same work.

These are results of a first version: one laptop CPU, Python and NumPy, small tasks, hyperparameters not tuned, 3 to
8 seeds.
