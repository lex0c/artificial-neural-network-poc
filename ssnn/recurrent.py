import numpy as np

from activations import sigmoid, softmax
from ssnn.gradients import sequence_gradients
from ssnn.model import PARAMETER_NAMES


# Recurrent models for sequences, each with the same interface for ssnn.training.train_sequences:
#   params, names   the parameters (a dict) and the order the optimizer sees them in
#   forward(inputs) the probabilities of the classes after the last event of the sequence, and a cache
#   gradients(cache, label)   the loss on the last event, -log p_T[label], and its gradient by backpropagation through
#                   time, as a dict of arrays shaped like params
#   touched         how many numbers it touches per event
# Every one starts each sequence from the zero state, and its decoder reads concat(x_t, state), like the SSNN's.


def init(rng, shape, fan_in, dtype):
    return np.ascontiguousarray(rng.normal(0, 1 / np.sqrt(fan_in), shape), dtype=dtype)


# A dense recurrent network: h_t = tanh(W concat(x_t, h_(t-1)) + b)
class TanhRNN:
    names = ("W", "b", "W_y", "b_y")

    def __init__(self, input_dim, hidden_dim, num_classes, seed=0, dtype="float32"):
        rng, fan_in = np.random.default_rng(seed), input_dim + hidden_dim
        self.params = {"W": init(rng, (hidden_dim, fan_in), fan_in, dtype), "b": np.zeros(hidden_dim, dtype),
                       "W_y": init(rng, (num_classes, fan_in), fan_in, dtype), "b_y": np.zeros(num_classes, dtype)}
        self.d, self.dtype = input_dim, np.dtype(dtype)
        self.touched = sum(array.size for array in self.params.values())

    def forward(self, inputs):
        p = self.params
        h = np.zeros(len(p["b"]), self.dtype)
        steps = []

        for x in np.asarray(inputs, self.dtype):
            xh = np.concatenate([x, h])
            h = np.tanh(p["W"] @ xh + p["b"])
            steps.append((xh, h))

        decoder_input = np.concatenate([x, h])
        return softmax(p["W_y"] @ decoder_input + p["b_y"]), (steps, decoder_input)

    def gradients(self, cache, label):
        steps, decoder_input = cache
        p, d = self.params, self.d
        grads = {name: np.zeros_like(array) for name, array in p.items()}
        delta, loss = output_delta(softmax(p["W_y"] @ decoder_input + p["b_y"]), label)
        grads["W_y"] += np.outer(delta, decoder_input)
        grads["b_y"] += delta
        dh = p["W_y"][:, d:].T @ delta

        for xh, h in reversed(steps):
            d_u = dh * (1 - h ** 2)
            grads["W"] += np.outer(d_u, xh)
            grads["b"] += d_u
            dh = (p["W"].T @ d_u)[d:]  # every event transforms the whole state: the gradient goes through all of them

        return loss, grads


# A gated recurrent unit (GRU): an update gate z decides how much of the state to keep at each event, which lets it
# hold a value across many events (z close to 1), and a reset gate r how much of it the candidate reads.
#   z = sigmoid(W_z concat(x, h) + b_z),  r = sigmoid(W_r concat(x, h) + b_r)
#   n = tanh(W_n concat(x, r * h) + b_n),  h_t = (1 - z) n + z h
# update_bias is the starting value of b_z. With 0, z starts around 0.5 and the gradient of an old state halves at every
# event; a positive one, e.g. 2 (z around 0.88), makes the GRU keep its state by default, a common initialization for
# long memories.
class GRU:
    names = ("W_z", "b_z", "W_r", "b_r", "W_n", "b_n", "W_y", "b_y")

    def __init__(self, input_dim, hidden_dim, num_classes, seed=0, dtype="float32", update_bias=0.0):
        rng, fan_in = np.random.default_rng(seed), input_dim + hidden_dim
        self.params = {}
        for gate in ("z", "r", "n"):
            self.params[f"W_{gate}"] = init(rng, (hidden_dim, fan_in), fan_in, dtype)
            self.params[f"b_{gate}"] = np.full(hidden_dim, update_bias if gate == "z" else 0.0, dtype)
        self.params["W_y"] = init(rng, (num_classes, fan_in), fan_in, dtype)
        self.params["b_y"] = np.zeros(num_classes, dtype)
        self.d, self.dtype = input_dim, np.dtype(dtype)
        self.touched = sum(array.size for array in self.params.values())

    def forward(self, inputs):
        p = self.params
        h = np.zeros(len(p["b_z"]), self.dtype)
        steps = []

        for x in np.asarray(inputs, self.dtype):
            xh = np.concatenate([x, h])
            z = sigmoid(p["W_z"] @ xh + p["b_z"])
            r = sigmoid(p["W_r"] @ xh + p["b_r"])
            xrh = np.concatenate([x, r * h])
            n = np.tanh(p["W_n"] @ xrh + p["b_n"])
            steps.append((xh, xrh, h, z, r, n))
            h = (1 - z) * n + z * h

        decoder_input = np.concatenate([x, h])
        return softmax(p["W_y"] @ decoder_input + p["b_y"]), (steps, decoder_input)

    def gradients(self, cache, label):
        steps, decoder_input = cache
        p, d = self.params, self.d
        grads = {name: np.zeros_like(array) for name, array in p.items()}
        delta, loss = output_delta(softmax(p["W_y"] @ decoder_input + p["b_y"]), label)
        grads["W_y"] += np.outer(delta, decoder_input)
        grads["b_y"] += delta
        dh = p["W_y"][:, d:].T @ delta

        for xh, xrh, h_prev, z, r, n in reversed(steps):
            d_n = dh * (1 - z)
            d_z = dh * (h_prev - n)
            dh_prev = dh * z

            d_un = d_n * (1 - n ** 2)
            grads["W_n"] += np.outer(d_un, xrh)
            grads["b_n"] += d_un
            d_rh = (p["W_n"].T @ d_un)[d:]
            dh_prev += d_rh * r

            d_uz = d_z * z * (1 - z)
            grads["W_z"] += np.outer(d_uz, xh)
            grads["b_z"] += d_uz
            dh_prev += (p["W_z"].T @ d_uz)[d:]

            d_ur = d_rh * h_prev * r * (1 - r)
            grads["W_r"] += np.outer(d_ur, xh)
            grads["b_r"] += d_ur
            dh_prev += (p["W_r"].T @ d_ur)[d:]

            dh = dh_prev

        return loss, grads


# The SSNN with the same interface: forward runs model.step over the sequence from the zero state
class SequenceSSNN:
    names = PARAMETER_NAMES

    def __init__(self, model):
        self.model, self.params = model, model.params
        c = model.config
        expert = c["state_dim"] * (c["input_dim"] + c["state_dim"]) + c["state_dim"]
        router = 0 if c["router"] != "learned" else model.params["W_r"].size + model.params["b_r"].size
        self.touched = c["top_k"] * expert + router + model.params["W_y"].size + model.params["b_y"].size

    def forward(self, inputs):
        state, traces = self.model.new_state(), []
        for x in inputs:
            probabilities, state, trace = self.model.step(x, state)
            traces.append(trace)
        return probabilities, traces

    def gradients(self, traces, label):
        return sequence_gradients(self.model, traces, label)

    # The router logits and inputs of every event, for the balance loss of a minibatch (None without a learned router)
    def router_events(self, traces):
        if self.model.config["router"] != "learned":
            return None
        return np.array([trace["router_logits"] for trace in traces]), np.array([trace["q"] for trace in traces])


def output_delta(probabilities, label):
    delta = probabilities.copy()
    delta[label] -= 1
    return delta, -np.log(max(float(probabilities[label]), 1e-12))


# The hidden size of a GRU that touches about target numbers per event: 3 (d + h + 1) h + C (d + h + 1)
def gru_hidden_for(target, input_dim, num_classes):
    # 3 h^2 + (3 (d + 1) + C) h + C (d + 1) - target = 0
    a, b, c = 3, 3 * (input_dim + 1) + num_classes, num_classes * (input_dim + 1) - target
    return max(1, round((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)))
