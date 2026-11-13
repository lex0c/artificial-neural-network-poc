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
python -m ssnn.scripts.train_memory --seeds 7       # experiment B: about 30 minutes per seed
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
  B, it did better than the selected-only gradient in one of four comparisons (the one-hot input with training gaps up
  to 100, within 200 events), and worse in the other three.
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

Registers, each holding one of a few values. A sequence writes every register, then has a preamble of 0 to 30 events
like those of a gap, then writes the register it will ask for with a new value, then has a gap of NOOP events, with a
rewrite of another register now and then (10% of the events), and ends with a QUERY of that register: the answer is
the last value written to it. The rewrites of the preamble (of any register) and the new value make registers
replace their values, and the last write of the queried register can come at any position, so nothing but the QUERY
tells which value to answer: a model has to keep the last value of every register. The NOOPs carry 8 noise features
that don't matter, which a recurrent network still has to keep out of its memory. Two settings:

- **8 registers of one bit, one-hot**: each pair of register and value, and each QUERY, is an input feature of its own,
  so nothing is common to the writes and the QUERY of a register. A guess is right half the time.
- **16 registers of 4 values, keyed**: the register is an input feature shared by its writes and its QUERY, like the
  key of a memory, and the value and the operation are others. A guess is right a quarter of the time.

The models train on 2,000 sequences with distances (events from the last write of the queried register to the QUERY)
up to 100, and are tested on distances up to 500, which they never saw:

- **SSNN, oracle**: one expert per register and one for the NOOPs, chosen by hand (`router: "fixed"`, one expert per
  event): what the architecture can do when the routing is right.
- **SSNN, gated write**: 16 experts (32 with the keyed input), 2 per event, the router learns which, trained by
  backpropagation through time. Each expert writes in proportion to the weight the router gives it (`write: "gated"`,
  `h = (1 - rho a) h + rho a n`), so a small weight leaves a state almost unchanged and the router gets a gradient
  through the write.
- **SSNN, gated, dense router gradient**: the same, with the dense router gradient (see the design choices).
- **GRUs**: gated recurrent units with as much work per event as the learned SSNN (28 neurons one-hot, 32 keyed), and
  with 64. The bias of their update gate starts at 2 (`--gru-update-bias 2`), so they start by keeping most of their
  state at every event, the usual initialization for long memories.

Every model trains 40 epochs at learning rates 0.001 and 0.003, and keeps, on each seed, the learning rate and the
epoch with the lowest validation loss; seeds 7, 19 and 43. Each run of one model, at one learning rate, on one seed,
took 15 to 75 minutes, 12 at a time:

```sh
python -m ssnn.scripts.train_memory --registers 16 --values 4 --encoding keyed --experts 32 --max-distance 100 \
    --epochs 40 --learning-rates 0.001 0.003 --gru-update-bias 2 --models oracle gated dense gru-same-work gru-64
python -m ssnn.scripts.train_memory --max-distance 100 --epochs 40 --learning-rates 0.001 0.003 --gru-update-bias 2 \
    --models oracle gated dense gru-same-work gru-64
```

Mean test accuracy (worst seed), and the validation loss, where a constant answer gives ln 4 = 1.386 with 4 values
and ln 2 = 0.693 with 2:

16 registers of 4 values, keyed:

| Model | Numbers touched per event | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 | Validation loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SSNN, oracle | 2,304 | 99.6% (99.3) | 99.3% (99.0) | 99.7% (99.3) | 99.6% (99.3) | 100% | 0.016 |
| SSNN, gated write | 6,400 | 76.0% (73.7) | 72.8% (71.7) | 66.4% (62.3) | 62.2% (62.0) | 50.9% (45.3) | 0.828 |
| SSNN, gated, dense router gradient | 6,400 | 65.8% (63.7) | 62.6% (59.7) | 55.2% (49.0) | 50.6% (39.3) | 37.4% (34.0) | 1.038 |
| GRU (32), same work | 6,400 | 58.8% (57.3) | 54.4% (52.7) | 50.9% (49.7) | 47.7% (46.3) | 45.2% (38.3) | 1.106 |
| GRU (64) | 18,816 | 58.3% (53.3) | 54.1% (53.7) | 50.0% (46.7) | 49.6% (47.0) | 44.7% (42.7) | 1.119 |

8 registers of 2 values, one-hot:

| Model | Numbers touched per event | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 | Validation loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SSNN, oracle | 2,244 | 100% | 100% | 100% | 100% | 100% | 0.002 |
| SSNN, gated write | 5,412 | 76.2% (73.0) | 74.3% (67.3) | 70.7% (64.7) | 65.3% (56.3) | 62.9% (54.3) | 0.507 |
| SSNN, gated, dense router gradient | 5,412 | 81.0% (72.0) | 80.9% (73.7) | 78.6% (73.3) | 70.9% (69.7) | 59.3% (54.7) | 0.477 |
| GRU (28), same work | 5,332 | 61.0% (60.0) | 57.7% (57.0) | 55.2% (51.3) | 53.8% (53.0) | 52.0% (49.7) | 0.684 |
| GRU (64) | 19,012 | 58.6% (57.0) | 58.8% (56.7) | 52.9% (51.0) | 56.7% (55.0) | 53.7% (51.7) | 0.688 |

What it shows:

- **With the right routing, the architecture is a perfect memory**: the oracle answers 99 to 100% at every distance,
  up to 5 times longer than any in its training, with the least work per event of all the models. Its memory experts
  don't run during the gaps, and a rewrite of another register goes to the expert of that register, so nothing
  disturbs them; and their gradient reaches the write without fading (a test checks it is the same after 50 or 200
  NOOPs).
- **The learned SSNN is far from it**: with the keyed input, 76% from 1 to 10 events, falling with the distance to 51%
  from 201 to 500; the best it does on any setting is 92% from 1 to 10 events (one-hot, trained on gaps up to 30, in
  the next section). The oracle has the same experts, integrator and decoder; what separates them is the routing. A
  learned router doesn't find one that gives each register experts its writes and its QUERY share, even when the
  input names the register: a routing that is a linear function of the input, which the router can represent.
- **Still, it beats the GRUs on both settings**: with the keyed input, the gated SSNN is above the GRU of the same work
  on every seed and at every distance, by 14 to 18 points on average up to 200 events and by 6 from 201 to 500, and
  its worst seed is above the best seed of the GRU up to 200 events. With the one-hot input, it is ahead on every seed
  and at every distance too, by 11 to 17 points on average. The GRU with 3 times the work does no better than the one
  of the same work.
- **The GRUs memorize their training sequences instead of learning to remember**: in a run that printed it (the GRU
  of the same work, keyed, seed 7, a diagnosis outside the reports), the training loss fell to 0.07 by the 40th epoch
  while the validation loss rose from the 10th, which is why the GRUs keep the parameters of their 7th to 12th epoch.
  It isn't only a lack of data: with 4 times as many training sequences (8,000, 20 epochs, both learning rates) the
  GRU no longer memorized them, and its validation loss went down (0.92 against 1.10), but it still answered only 51
  to 65% of the tests right at every distance, and 61 to 63% of its training sequences. With the bias of its update
  gate at 0, it didn't leave chance.
- **The dense router gradient helps with the one-hot input and hurts with the keyed one**: 5 to 8 points above the
  selected-only gradient up to 200 events with the one-hot input (with a lower validation loss on 2 seeds of 3), 4
  below from 201 to 500, and 10 to 14 points below at every distance with the keyed one, as on the first version of
  the task.
- **The router still lets the noise into the memory**: the NOOPs of the gaps of 101 to 200 events select one of the
  experts the QUERY reads 13% of the time with the keyed input and 24% with the one-hot one (none with the oracle).

### Training on gaps up to 30

The same comparison, with the training and validation sequences at distances up to 30 (`--max-distance 30`) instead of
100: shorter sequences, and no gap in the training as long as most of the tests. Mean test accuracy (worst seed):

| Model | 1 to 10 | 11 to 30 | 31 to 100 | 101 to 200 | 201 to 500 |
| --- | --- | --- | --- | --- | --- |
| **16 registers of 4 values, keyed** | | | | | |
| SSNN, oracle | 99.4% (98.3) | 99.3% (99.3) | 99.9% (99.7) | 99.7% (99.7) | 99.4% (99.0) |
| SSNN, gated write | 72.9% (65.7) | 67.3% (60.7) | 43.0% (33.7) | 31.4% (26.3) | 27.2% (24.7) |
| SSNN, gated, dense router gradient | 65.8% (59.0) | 56.8% (52.0) | 32.7% (28.7) | 26.7% (25.3) | 23.0% (21.3) |
| GRU (32), same work | 62.3% (60.7) | 55.9% (54.0) | 31.2% (24.3) | 27.3% (24.3) | 26.0% (23.7) |
| GRU (64) | 61.1% (57.0) | 56.8% (55.0) | 35.2% (26.0) | 29.8% (24.0) | 26.7% (23.7) |
| **8 registers of 2 values, one-hot** | | | | | |
| SSNN, oracle | 99.9% (99.7) | 100% | 99.9% (99.7) | 100% | 100% |
| SSNN, gated write | 92.1% (90.3) | 86.6% (84.3) | 67.1% (62.0) | 54.1% (45.3) | 53.3% (51.3) |
| SSNN, gated, dense router gradient | 86.4% (84.3) | 80.4% (77.0) | 65.3% (62.7) | 51.6% (49.0) | 52.8% (48.3) |
| GRU (28), same work | 73.0% (69.3) | 67.9% (65.3) | 50.0% (49.7) | 51.9% (50.3) | 50.8% (47.7) |
| GRU (64) | 73.9% (72.7) | 67.0% (63.7) | 51.9% (44.7) | 52.9% (51.3) | 50.1% (50.0) |

- **Only the oracle remembers beyond the distances it trained on**: 99 to 100% up to 500 events, 16 times the longest
  gap of its training, while every learned model falls to about chance from 101 events on. Training on gaps up to 100
  is what made the learned models keep part of their memory there (51% and 63% from 201 to 500 for the gated SSNN).
- **Within the trained distances, the shorter gaps helped with the one-hot input, not with the keyed one**: with the
  one-hot input, the gated SSNN went from 76% to 92% from 1 to 10 events, and from 74% to 87% from 11 to 30, 19 points
  above the GRU of the same work on average and 14 to 23 on each seed; with the keyed one, it did slightly worse (73%
  against 76% from 1 to 10), 11 points above the GRU of the same work on average, but only 2 to 6 points on two seeds
  of three (24 to 26 on the third).
- **The dense router gradient is below the selected-only one on both settings**, by 6 to 7 points from 1 to 10 events.

These rows give the commit as `056da1d-dirty`: they ran while only this README and `reports/memory.csv` differed from
that commit.

### The first version of the task

These results replace those of a first version of the task, in which the last write of the queried register always
came right after the first writes of every register, and repeated the value of its first write: a register never
replaced a value. A rule that ignores the QUERY, the value of the last write before the first NOOP, was right 89 to
97% of the time on its test sets (28% with 4 values and 52% with 2 on the corrected task, about chance). The trained
models used that, each its own way: the GRUs learned the rule, and fell to its level when the position of that write
was moved; the learned SSNN counted the events, through its global state, to find that position, and sent that write
to experts of its own; the oracle kept the first value of each register, and never learned to replace it (on
sequences where the queried register was rewritten after its first write, it answered right 0 to 8% of the time).

So the comparisons between learned models on that version said nothing about memory: the learned SSNN beating the
GRU of the same work up to 100 events, the effect of the input encoding, the seeds that seemed to find the routing of
the oracle, the memory fading beyond the trained distances. Its rows stay in `reports/memory.csv` (`preamble` 0,
`new_value` False), and `--preamble 0 --same-value` gives that version back, sequence for sequence.

## Status of the hypotheses

| Hypothesis | Status |
| --- | --- |
| Running k of E experts reduces the computation per event | supported: the cost per event barely depends on E, 30 times less than running all 128 experts |
| That sparsity makes inference faster on a CPU, at a similar quality | not supported at this scale: 2 to 4 times slower than a dense RNN doing the same work per event in Python |
| The stored capacity buys quality | not supported in experiment D, where the SSNN stays below dense networks of the same work; the task turned out to be limited by data, so a larger version is still open |
| Persistent states help on tasks that need memory | supported with the right routing: chosen by hand, a perfect memory of 8 or 16 registers at every distance up to 500 events, with the least work per event, even when trained on gaps up to 30. With a learned router, the SSNN beats GRUs of the same work and of 3 times the work within the distances it trained on, on both settings and both training distances, by 10 to 20 points on average, but none of the learned models solves the task: the SSNN stays 8 to 49 points below the oracle, and loses most of its memory beyond the distances it trained on |
| A learned routing specializes the experts | supported in experiments A and D (far above random routing, with regime information well above its bias), with partial specialization; not enough in experiment B, where the router doesn't find the routing of the oracle, one that gives each register experts its writes and its QUERY share, even when the input names the register: that routing is worth 8 to 49 points there |

## Conclusion so far

The mechanism works: the gradients are right, the selective execution matches the dense one, the router learns and
specializes the experts, and the stored capacity comes almost for free in latency.

Experiments A and D (independent examples) and the latency benchmark go against the SSNN: against a dense network
doing the same work per event, it loses on accuracy and on speed. Its stored capacity didn't buy quality, and the
specialization even cost accuracy in experiment D, as each expert learns from only part of the data.

Experiment B (memory) shows what the architecture is for, and what it lacks. Experts that don't run keep their state
exactly: with the routing chosen by hand, the SSNN is a perfect memory of 8 registers of one bit or of 16 registers of
4 values, at every distance up to 500 events, with less work per event than any other model, even when it trained on
gaps up to 30. With a learned router, it beats GRUs of the same work, and of 3 times the work, within the distances it
trained on, by 10 to 20 points on average. But that is a contest between models that don't solve the task: the GRUs
memorize their training sequences instead of learning to remember, and 4 times as much data didn't change their
accuracy much; the learned SSNN answers at best 76% (16 registers of 4 values) and 92% (8 of one bit) right from 1 to
10 events, and 51% and 63% from 201 to 500, where the oracle answers 99 to 100%; and trained on gaps up to 30, every
learned model falls to about chance from 101 events on, while the oracle doesn't. What falls short is the router: it
doesn't find the routing the oracle uses, even when the input names the register and that routing is a linear
function of the input.

A first version of the task let every learned model answer without remembering (see
[The first version of the task](#the-first-version-of-the-task)). Its results are withdrawn, among them that the
learned SSNN found the routing of the oracle on some seeds, and that it beat the GRU of the same work up to 100 events
when the input names the register.

So the SSNN is the better memory at the same work per event only in a weak sense: neither it nor the GRU solves the
task, and it fails less. What stays open, in the order it would be worth testing:

1. **Why the learned router doesn't find the routing of the oracle**: where the writes and the QUERY of a register
   go, and whether a change aimed at that (a router that reads the input alone, without the global state, or a
   training signal that makes the QUERY read the experts of the last write of its register) closes part of the 8 to 49
   points.
2. **Whether a dense network of the same work can learn this task at all**: the GRUs with longer trainings, other
   sizes or a schedule of the learning rate, and other dense memories (an LSTM, attention over the past events).
3. **Capacity at a larger scale**, and **speed** (compiled code, routing with locality, grouping the events of many
   sessions by expert), which matters only where the SSNN is better: in Python, an SSNN event still takes 2 to 4
   times as long as a dense network doing the same work.

These are results of a first version: one laptop CPU, Python and NumPy, small tasks, hyperparameters not tuned, 3
seeds.
