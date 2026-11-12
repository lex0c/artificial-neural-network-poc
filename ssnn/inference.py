import math
import time

import numpy as np

from ssnn.model import State


# Lean inference engines, one event at a time, for the latency benchmark (scripts/bench_cpu.py). model.step copies the
# state and keeps a trace for the gradients, which is right for training and for catching bugs, but copying the whole
# state of every expert at each event (128 KiB with 128 experts of 256 values) would hide what the architecture costs.
# These update the state in place, reuse their buffers and return only the probabilities. The tests check that they
# give the same results as model.step.
#
# Each engine has step(x, state) and the numbers it touches per event, so the benchmark can compare them fairly.


def stable_softmax(z, out):
    np.subtract(z, z.max(), out=out)
    np.exp(out, out=out)
    out /= out.sum()
    return out


# The SSNN as it is meant to run: the router, then only the k selected experts. Two choices make it cheaper than a
# direct translation of the equations, without changing the result:
#   - the input and recurrent weights of each expert are packed side by side, W_xh[i] = [W_x[i] | W_h[i]], so an
#     expert is one matrix-vector product with concat(x, h_i), with no temporary arrays (a copy of the expert weights,
#     made once)
#   - the softmax of the k selected logits is computed with Python floats: for a few numbers, cheaper than NumPy calls
# marks, when given, gets the time in nanoseconds after each part (router, top-k, experts, integrator, decoder), for the
# profile of the benchmark.
class SelectiveEngine:
    def __init__(self, model):
        p, c = model.params, model.config
        self.W_xh = np.ascontiguousarray(np.concatenate([p["W_x"], p["W_h"]], axis=2))  # (E, H, d + H)
        self.b, self.W_r, self.b_r, self.W_y, self.b_y = (p[name] for name in ("b", "W_r", "b_r", "W_y", "b_y"))
        self.E, self.d, self.k = c["num_experts"], c["input_dim"], c["top_k"]
        self.rho, self.beta, self.tau = c["rho"], c["beta"], c["routing_temperature"]
        self.q = np.empty(c["input_dim"] + c["state_dim"], model.dtype)   # concat(x, g), reused
        self.xh = np.empty(c["input_dim"] + c["state_dim"], model.dtype)  # concat(x, h_i), reused
        self.u = np.empty(c["state_dim"], model.dtype)
        self.v = np.empty(c["state_dim"], model.dtype)
        self.probabilities = np.empty(c["num_classes"], model.dtype)
        self.touched = (self.k * (self.W_xh[0].size + self.b[0].size)
                        + self.W_r.size + self.b_r.size + self.W_y.size + self.b_y.size)

    def step(self, x, state, marks=None):
        d, q, xh, u, v = self.d, self.q, self.xh, self.u, self.v
        q[:d] = x
        q[d:] = state.g
        logits = self.W_r @ q
        logits += self.b_r
        if marks is not None:
            marks.append(time.perf_counter_ns())

        # The k largest logits, in decreasing order, ties broken by the lower index (as in model.route). Up to about a
        # thousand experts, a stable sort of all the logits is cheaper than np.argpartition followed by sorting the k
        # it keeps (about 4 against 7.5 microseconds with 128)
        if self.k == 1:
            selected = [int(np.argmax(logits))]  # the first of the largest
        elif self.E <= 1024:
            selected = np.argsort(-logits, kind="stable")[:self.k].tolist()
        else:
            candidates = np.sort(np.argpartition(logits, self.E - self.k)[self.E - self.k:])
            selected = candidates[np.argsort(-logits[candidates], kind="stable")].tolist()

        values = [float(logits[i]) / self.tau for i in selected]
        largest = max(values)
        exps = [math.exp(value - largest) for value in values]
        total = sum(exps)
        if marks is not None:
            marks.append(time.perf_counter_ns())

        xh[:d] = x
        for j, i in enumerate(selected):
            h = state.h[i]  # a view: the state of expert i, updated in place
            xh[d:] = h
            np.matmul(self.W_xh[i], xh, out=u)
            u += self.b[i]
            np.tanh(u, out=u)
            u *= self.rho
            h *= 1 - self.rho
            h += u  # h = (1 - rho) h + rho n
            if j == 0:
                np.multiply(h, exps[j] / total, out=v)
            else:
                v += (exps[j] / total) * h
        if marks is not None:
            marks.append(time.perf_counter_ns())

        g = state.g
        g *= 1 - self.beta
        v *= self.beta
        g += v
        if marks is not None:
            marks.append(time.perf_counter_ns())

        q[d:] = g  # concat(x, g) again, for the decoder
        probabilities = stable_softmax(self.W_y @ q + self.b_y, self.probabilities)
        if marks is not None:
            marks.append(time.perf_counter_ns())
        return probabilities


# The same architecture running all of its experts, in a few large operations instead of a loop: what the SSNN does
# with k = E, written the way a dense implementation would be
class AllExpertsEngine:
    def __init__(self, model):
        p, c = model.params, model.config
        E, H, d = c["num_experts"], c["state_dim"], c["input_dim"]
        self.W_x = p["W_x"].reshape(E * H, d)  # all the input weights as one matrix
        self.W_h, self.b, self.W_r, self.b_r, self.W_y, self.b_y = (p[name] for name in ("W_h", "b", "W_r", "b_r", "W_y", "b_y"))
        self.E, self.H, self.d = E, H, d
        self.rho, self.beta, self.tau = c["rho"], c["beta"], c["routing_temperature"]
        self.q = np.empty(d + H, model.dtype)
        self.weights = np.empty(E, model.dtype)
        self.probabilities = np.empty(c["num_classes"], model.dtype)
        self.touched = sum(array.size for array in p.values())

    def step(self, x, state):
        d, q = self.d, self.q
        q[:d] = x
        q[d:] = state.g
        stable_softmax((self.W_r @ q + self.b_r) / self.tau, self.weights)

        u = (self.W_x @ x).reshape(self.E, self.H)
        u += np.matmul(self.W_h, state.h[:, :, None])[:, :, 0]  # each expert's own recurrent matrix
        u += self.b
        np.tanh(u, out=u)
        state.h *= 1 - self.rho
        state.h += self.rho * u

        state.g *= 1 - self.beta
        state.g += self.beta * (self.weights @ state.h)
        q[d:] = state.g
        return stable_softmax(self.W_y @ q + self.b_y, self.probabilities)


# A dense recurrent network, the usual alternative: h = tanh(W [x, h] + b), p = softmax(W_y [x, h] + b_y). Its hidden
# size is chosen to touch as many numbers per event as the SSNN (the same work), or to have as many in total (the same
# stored capacity).
class DenseRNN:
    def __init__(self, input_dim, hidden_dim, num_classes, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        fan_in = input_dim + hidden_dim
        self.W = rng.normal(0, 1 / np.sqrt(fan_in), (hidden_dim, fan_in)).astype(dtype)
        self.b = np.zeros(hidden_dim, dtype)
        self.W_y = rng.normal(0, 1 / np.sqrt(fan_in), (num_classes, fan_in)).astype(dtype)
        self.b_y = np.zeros(num_classes, dtype)
        self.d = input_dim
        self.xh = np.empty(fan_in, dtype)  # concat(x, h), reused
        self.probabilities = np.empty(num_classes, dtype)
        self.touched = self.W.size + self.b.size + self.W_y.size + self.b_y.size

    def new_state(self):
        return State(None, np.zeros(len(self.b), self.b.dtype))  # only a hidden vector, kept in g

    def step(self, x, state):
        xh = self.xh
        xh[:self.d] = x
        xh[self.d:] = state.g
        h = self.W @ xh
        h += self.b
        np.tanh(h, out=state.g)
        xh[self.d:] = state.g
        return stable_softmax(self.W_y @ xh + self.b_y, self.probabilities)


# The hidden size of a dense RNN that touches about target numbers per event: (d + h + 1) * h + (d + h + 1) * C
def rnn_hidden_for(target, input_dim, num_classes):
    # h^2 + (d + 1 + C) h + C (d + 1) - target = 0
    b, c = input_dim + 1 + num_classes, num_classes * (input_dim + 1) - target
    return max(1, round((-b + np.sqrt(b * b - 4 * c)) / 2))
