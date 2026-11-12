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


# Experiment B: memory. A set of hidden registers, each holding one of `values` values (2 by default: one bit). A
# sequence first writes every register, then the
# register it will ask for once more, then has a gap of NOOP events, with a rewrite of another register now and then,
# and ends with a QUERY of that register, whose answer is the last value written to it. The NOOPs carry noise features
# that don't matter, which a dense recurrent network still has to keep out of its state. The distance is the number of
# events from the last write of the queried register to the QUERY.
#
# The inputs: one-hot over the commands, SET(i, v) for each register and value, QUERY(i) and NOOP, followed by the
# noise features (non-zero only on the NOOPs). The answer is one of the values: a class. With 1 register of one bit and
# no noise, it is a single bit to remember.
class MemoryTask:
    def __init__(self, registers=8, noise=8, rewrite_rate=0.1, values=2):
        self.registers, self.noise, self.rewrite_rate, self.values = registers, noise, rewrite_rate, values
        self.input_dim = (values + 1) * registers + 1 + noise
        self.noop_index = (values + 1) * registers

    def set_event(self, register, value):
        x = np.zeros(self.input_dim, np.float32)
        x[self.values * register + value] = 1
        return x

    def query_event(self, register):
        x = np.zeros(self.input_dim, np.float32)
        x[self.values * self.registers + register] = 1
        return x

    def noop_event(self, rng):
        x = np.zeros(self.input_dim, np.float32)
        x[self.noop_index] = 1
        x[self.noop_index + 1:] = rng.normal(0, 1, self.noise)
        return x

    def sequence(self, rng, distance):
        values = rng.integers(0, self.values, self.registers)
        events = [self.set_event(i, values[i]) for i in rng.permutation(self.registers)]
        target = int(rng.integers(0, self.registers))
        events.append(self.set_event(target, values[target]))  # its last write, distance events before the QUERY
        others = [i for i in range(self.registers) if i != target]

        for _ in range(distance - 1):
            if others and rng.random() < self.rewrite_rate:
                other = int(rng.choice(others))
                values[other] = rng.integers(0, self.values)
                events.append(self.set_event(other, values[other]))
            else:
                events.append(self.noop_event(rng))

        events.append(self.query_event(target))
        return np.array(events), int(values[target]), distance

    # num sequences whose distance is drawn uniformly between low and high (included): (inputs, label, distance) each
    def examples(self, num_sequences, distances, rng):
        return [self.sequence(rng, int(distance)) for distance in rng.integers(distances[0], distances[1] + 1, num_sequences)]

    # The routing an oracle would choose, for a fixed router with k = 1: expert i for the writes and the QUERY of
    # register i, and one more expert for the NOOPs (whatever their largest feature)
    def oracle_routes(self):
        return ([[i // self.values] for i in range(self.values * self.registers)] + [[i] for i in range(self.registers)]
                + [[self.registers]] * (1 + self.noise))


# Training and validation with distances up to 30 (or max_distance); tests with distances up to 30 and beyond, which
# the training may never have seen: 31 to 100, 101 to 200 and 201 to 500. Each range has a random generator of its
# own, so adding one doesn't change the others.
MEMORY_TEST_DISTANCES = ((1, 10), (11, 30), (31, 100), (101, 200), (201, 500))


def experiment_memory(seed, task, sizes=(2000, 500), test_size=300, max_distance=30):
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 2]).spawn(2 + len(MEMORY_TEST_DISTANCES))]
    train = task.examples(sizes[0], (1, max_distance), generators[0])
    val = task.examples(sizes[1], (1, max_distance), generators[1])
    tests = {distances: task.examples(test_size, distances, rng)
             for distances, rng in zip(MEMORY_TEST_DISTANCES, generators[2:])}
    return train, val, tests
