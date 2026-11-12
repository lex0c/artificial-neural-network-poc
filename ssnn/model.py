import json

import numpy as np

from activations import softmax  # numerically stable: subtracts the maximum before exp
from ssnn.config import validate


# The SSNN, version 0. For each event x_t:
#
#   router      q = concat(x_t, g_(t-1));  z = W_r q + b_r;  S = the k largest z;  a = softmax(z[S] / tau)
#   experts     for i in S:  u_i = W_x[i] x_t + W_h[i] h_i + b[i];  n_i = tanh(u_i);  h_i = (1 - s_i) h_i + s_i n_i
#               with s_i = rho (write "fixed") or rho a_i (write "gated"); the other experts keep their state unchanged
#   integrator  v = sum over S of a_i h_i;  g_t = (1 - beta) g_(t-1) + beta v
#   decoder     p = softmax(W_y concat(x_t, g_t) + b_y);  prediction = argmax(p)
#
# Dimensions: d inputs, E experts with a state of H values each, a global state of H values, C classes.

PARAMETER_NAMES = ("W_x", "W_h", "b", "W_r", "b_r", "W_y", "b_y")


# The state of one session: h, one row per expert, and g, the global state. Sessions never share it: only the trained
# weights are common to all of them.
class State:
    def __init__(self, h, g):
        self.h = h  # (E, H)
        self.g = g  # (H,)


class SSNN:
    def __init__(self, config, seed=None):
        validate(config)
        self.config = config
        self.dtype = np.dtype(config["dtype"])
        E, H, d, C = config["num_experts"], config["state_dim"], config["input_dim"], config["num_classes"]
        seed = config["seed"] if seed is None else seed
        init_rng, self.routing_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))

        # Small weights scaled by the number of inputs of each product, zero biases
        def weights(shape, fan_in):
            return np.ascontiguousarray(init_rng.normal(0, 1 / np.sqrt(fan_in), shape), dtype=self.dtype)

        self.params = {
            "W_x": weights((E, H, d), d),          # each expert's matrix is contiguous in memory
            "W_h": weights((E, H, H), H),
            "b": np.zeros((E, H), self.dtype),
            "W_r": weights((E, d + H), d + H),
            "b_r": np.zeros(E, self.dtype),
            "W_y": weights((C, d + H), d + H),
            "b_y": np.zeros(C, self.dtype),
        }

    # The parameters as a list, in the order of PARAMETER_NAMES, for the optimizers (optimizers.py)
    def parameters(self):
        return [self.params[name] for name in PARAMETER_NAMES]

    def new_state(self):
        E, H = self.config["num_experts"], self.config["state_dim"]
        return State(np.zeros((E, H), self.dtype), np.zeros(H, self.dtype))

    # The k experts of an event and their weights. Learned: the k largest logits, ties broken by the lower index so
    # the choice is reproducible (np.argpartition doesn't keep a stable order). Random: k experts drawn at random, with
    # equal weights (a baseline that separates a learned routing from an arbitrary one). Fixed: the experts that
    # fixed_routes gives to the largest input of the event, with equal weights (an oracle that tests the rest of the
    # architecture when the routing is right).
    def route(self, q):
        k = self.config["top_k"]

        if self.config["router"] == "random":
            selected = self.routing_rng.choice(self.config["num_experts"], size=k, replace=False)
            return None, selected, np.full(k, 1 / k, self.dtype)

        if self.config["router"] == "fixed":
            selected = np.array(self.config["fixed_routes"][int(np.argmax(q[:self.config["input_dim"]]))])
            return None, selected, np.full(k, 1 / k, self.dtype)

        logits = self.params["W_r"] @ q + self.params["b_r"]
        selected = np.lexsort((np.arange(len(logits)), -logits))[:k]
        weights = softmax(logits[selected] / self.config["routing_temperature"])
        return logits, selected, weights

    # One event: returns the probability of each class, the new state (a copy: the given state isn't changed) and a
    # trace with what the router chose and what the backward pass needs (gradients.py)
    def step(self, x, state):
        p = self.params
        rho, beta = self.config["rho"], self.config["beta"]
        x = np.asarray(x, self.dtype)

        q = np.concatenate([x, state.g])
        logits, selected, weights = self.route(q)

        h = state.h.copy()
        h_prev = state.h[selected]  # (k, H), the states before this event
        n = np.empty_like(h_prev)

        for j, i in enumerate(selected):
            u = p["W_x"][i] @ x + p["W_h"][i] @ h_prev[j] + p["b"][i]
            n[j] = np.tanh(u)
            strength = write_strength(self.config, weights[j])
            h[i] = (1 - strength) * h_prev[j] + strength * n[j]

        v = weights @ h[selected]
        g = (1 - beta) * state.g + beta * v

        decoder_input = np.concatenate([x, g])
        probabilities = softmax(p["W_y"] @ decoder_input + p["b_y"])

        trace = {
            "x": x, "q": q, "router_logits": logits, "selected": selected, "weights": weights,
            "h_prev": h_prev, "n": n, "h_selected": h[selected], "decoder_input": decoder_input,
            "probabilities": probabilities,
        }
        return probabilities, State(h, g), trace

    # The parameters and the configuration, so a trained model can repeat its inferences
    def save(self, filepath):
        np.savez(filepath, config=json.dumps(self.config), **self.params)

    @classmethod
    def load(cls, filepath):
        with np.load(filepath) as saved:
            model = cls(json.loads(str(saved["config"])))
            for name in PARAMETER_NAMES:
                model.params[name] = saved[name].copy()
        return model


# How much of its new value a selected expert takes: rho, or rho times its weight with the gated write
def write_strength(config, weight):
    return config["rho"] * weight if config["write"] == "gated" else config["rho"]


# Parameter counts: all of them, and those an event touches (k experts, the router and the decoder)
def parameter_counts(config):
    E, H, d, C, k = (config[key] for key in ("num_experts", "state_dim", "input_dim", "num_classes", "top_k"))
    expert = H * d + H * H + H
    router, decoder = E * (d + H + 1), C * (d + H + 1)
    return {"total": E * expert + router + decoder, "per_event": k * expert + router + decoder}
