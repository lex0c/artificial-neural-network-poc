import numpy as np


# Experiment A: classification with distinct regimes. Each example has four numbers
# u1..u4 from a normal distribution and a regime m from 0 to 3, given as a one-hot vector, so x has d = 8 values. The
# binary label follows a different rule in each regime:
#
#   m = 0: u1 * u2 > 0       (u1 and u2 have the same sign)
#   m = 1: u1 + u2 > 0
#   m = 2: u3 * u4 > 0
#   m = 3: |u1| > |u2|
#
# The regime is given on purpose, so the routing can be tested: the task doesn't show that experts appear on their own.
NUM_REGIMES = 4


def regime_examples(num_examples, rng):
    u = rng.normal(0, 1, (num_examples, 4))
    regimes = rng.integers(0, NUM_REGIMES, num_examples)

    rules = np.stack([
        u[:, 0] * u[:, 1] > 0,
        u[:, 0] + u[:, 1] > 0,
        u[:, 2] * u[:, 3] > 0,
        np.abs(u[:, 0]) > np.abs(u[:, 1]),
    ], axis=1)
    labels = rules[np.arange(num_examples), regimes].astype(int)

    inputs = np.concatenate([u, np.eye(NUM_REGIMES)[regimes]], axis=1).astype(np.float32)
    return inputs, labels, regimes


# 12,000 examples to train, 3,000 to validate and 3,000 to test, each from its own random generator, so changing the
# size of one split doesn't change the others. Each split is (inputs, labels, regimes).
def experiment_a(seed, sizes=(12000, 3000, 3000)):
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    return [regime_examples(size, rng) for size, rng in zip(sizes, generators)]


# Experiment D: a task that needs capacity. R regimes, each with a rule of its own: a teacher, a small random network
# tanh(u W1_r^T) w2_r over the same features u, gives the label of its regime by the sign of its output (no biases, so
# the teacher is an odd function and its labels are balanced). The regime is given as a one-hot vector. The more
# regimes, the more different rules a model has to hold at once: a dense network with few hidden neurons has to share
# them between all the rules, while the SSNN can give each regime experts of its own.
class Teachers:
    def __init__(self, num_regimes, num_features=8, hidden=8, seed=0):
        rng = np.random.default_rng([seed, num_regimes])
        self.W1 = rng.normal(0, 1, (num_regimes, hidden, num_features)) / np.sqrt(num_features)
        self.w2 = rng.normal(0, 1, (num_regimes, hidden))
        self.num_regimes, self.num_features = num_regimes, num_features

    def labels(self, u, regimes):
        hidden = np.tanh(np.einsum("nf,nhf->nh", u, self.W1[regimes]))
        return (np.sum(hidden * self.w2[regimes], axis=1) > 0).astype(int)


def capacity_examples(num_examples, teachers, rng):
    u = rng.normal(0, 1, (num_examples, teachers.num_features))
    regimes = rng.integers(0, teachers.num_regimes, num_examples)
    inputs = np.concatenate([u, np.eye(teachers.num_regimes)[regimes]], axis=1).astype(np.float32)
    return inputs, teachers.labels(u, regimes), regimes


# The same number of examples per regime whatever the number of regimes (400 to train, 100 to validate and 100 to
# test by default), each split from its own random generator. Each split is (inputs, labels, regimes).
def experiment_capacity(seed, num_regimes, per_regime=(400, 100, 100)):
    teachers = Teachers(num_regimes, seed=seed)
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, num_regimes]).spawn(3)]
    return [capacity_examples(size * num_regimes, teachers, rng) for size, rng in zip(per_regime, generators)]
