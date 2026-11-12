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
