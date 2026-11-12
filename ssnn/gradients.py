import numpy as np

from activations import softmax
from ssnn.model import PARAMETER_NAMES, write_strength


# Gradients of one event, written by hand and checked numerically in the tests. The
# states before the event, h_(t-1) and g_(t-1), are treated as constants (a stop-gradient in time): enough for tasks
# whose answer depends on the current event, not for those that need to remember (that needs backpropagation through
# time, BPTT).


# The cross-entropy of one event, -log p[label], and its gradient with respect to every parameter, as a dict of
# arrays shaped like model.params. trace comes from model.step.
def task_gradients(model, trace, label):
    grads = {name: np.zeros_like(model.params[name]) for name in PARAMETER_NAMES}
    loss = add_task_gradients(model, trace, label, grads)
    return loss, grads


# The same, added in place to grads (e.g. the sum over a minibatch): only the rows of the selected experts and of the
# router change, so a training step doesn't create arrays the size of all the experts for every event. Returns the
# loss.
def add_task_gradients(model, trace, label, grads):
    params, config = model.params, model.config
    d = config["input_dim"]
    rho, beta, tau = config["rho"], config["beta"], config["routing_temperature"]

    probabilities = trace["probabilities"]
    loss = -np.log(max(float(probabilities[label]), 1e-12))

    # Decoder: softmax with cross-entropy gives dL/dlogits = p - one_hot(label)
    delta = probabilities.copy()
    delta[label] -= 1
    grads["W_y"] += np.outer(delta, trace["decoder_input"])
    grads["b_y"] += delta

    # Global state and integrator: l depends on g through W_y[:, d:], and g = (1 - beta) g_prev + beta v
    d_g = params["W_y"][:, d:].T @ delta
    d_v = beta * d_g

    # Experts: v = sum of a_j h_j, h_j = (1 - s_j) h_prev_j + s_j tanh(u_j), u_j = W_x x + W_h h_prev_j + b, where the
    # write strength s_j is rho, or rho a_j with the gated write (then a_j also changes how much h_j moves)
    selected, weights = trace["selected"], trace["weights"]
    d_weights = trace["h_selected"] @ d_v  # dL/da_j = d_v . h_j, through the read

    for j, i in enumerate(selected):
        d_h = weights[j] * d_v
        d_u = write_strength(config, weights[j]) * (1 - trace["n"][j] ** 2) * d_h
        grads["W_x"][i] += np.outer(d_u, trace["x"])
        grads["W_h"][i] += np.outer(d_u, trace["h_prev"][j])
        grads["b"][i] += d_u
        if config["write"] == "gated":
            d_weights[j] += rho * d_h @ (trace["n"][j] - trace["h_prev"][j])  # through the write

    # Router: a = softmax(z[S] / tau), so dL/dz_j = (a_j / tau) (dL/da_j - sum of a dL/da) for the selected experts,
    # and 0 for the others (the top-k choice itself isn't differentiable). With k = 1, a = [1] and the gradient is 0.
    if config["router"] == "learned":
        d_z = (weights / tau) * (d_weights - weights @ d_weights)
        grads["W_r"][selected] += np.outer(d_z, trace["q"])
        grads["b_r"][selected] += d_z

    return loss


# Backpropagation through time over a sequence of events of one session, with the loss on the last event only (the
# answer of a memory task): -log p_T[label]. traces are the traces of model.step for each event, in order, from the zero
# state. Returns the loss and its gradient with respect to every parameter.
#
# Going backwards, dh (one row per expert) and dg hold the gradient of the loss with respect to the states after the
# current event. At each event, only the selected experts compute something: an expert that didn't run kept its state
# unchanged (an identity), so its row of dh passes to the previous event as it is. The gradient of a state that waited
# many events between two uses therefore doesn't fade with the length of the wait, unlike in a dense recurrent network,
# which transforms its whole state at every event.
def sequence_gradients(model, traces, label):
    params, config = model.params, model.config
    d = config["input_dim"]
    rho, beta, tau = config["rho"], config["beta"], config["routing_temperature"]
    grads = {name: np.zeros_like(params[name]) for name in PARAMETER_NAMES}

    last = traces[-1]
    loss = -np.log(max(float(last["probabilities"][label]), 1e-12))
    delta = last["probabilities"].copy()
    delta[label] -= 1
    grads["W_y"] += np.outer(delta, last["decoder_input"])
    grads["b_y"] += delta

    dg = params["W_y"][:, d:].T @ delta  # dL/dg_T
    dh = np.zeros((config["num_experts"], config["state_dim"]), model.dtype)  # dL/dh_T: no loss on h directly

    for trace in reversed(traces):
        selected, weights = trace["selected"], trace["weights"]

        # g_t = (1 - beta) g_(t-1) + beta v_t, v_t = sum of a_j h_(t, j)
        d_v = beta * dg
        dg_prev = (1 - beta) * dg
        d_weights = trace["h_selected"] @ d_v

        for j, i in enumerate(selected):
            strength = write_strength(config, weights[j])
            d_h = dh[i] + weights[j] * d_v  # from the following events, and from v_t
            d_u = strength * d_h * (1 - trace["n"][j] ** 2)
            grads["W_x"][i] += np.outer(d_u, trace["x"])
            grads["W_h"][i] += np.outer(d_u, trace["h_prev"][j])
            grads["b"][i] += d_u
            if config["write"] == "gated":
                d_weights[j] += rho * d_h @ (trace["n"][j] - trace["h_prev"][j])  # through the write
            dh[i] = (1 - strength) * d_h + params["W_h"][i].T @ d_u  # dL/dh_(t-1, i)

        # The router read q = concat(x_t, g_(t-1)): its gradient reaches the global state of the previous event too
        if config["router"] == "learned":
            d_z = (weights / tau) * (d_weights - weights @ d_weights)
            grads["W_r"][selected] += np.outer(d_z, trace["q"])
            grads["b_r"][selected] += d_z
            dg_prev += (params["W_r"][selected].T @ d_z)[d:]

        dg = dg_prev

    return loss, grads


# Balance loss of a minibatch: L_bal = E * sum over the experts of (mean_pi_i - 1 / E)^2, where pi is the
# softmax over all the E router logits of each event and mean_pi its mean over the minibatch. It is 0 when the router
# gives every expert the same mean probability, and its gradient reaches the experts that weren't selected too. It
# encourages a balanced use, but doesn't guarantee balanced top-k counts.
#
# router_logits and router_inputs: one row per event of the minibatch (trace["router_logits"] and trace["q"]).
# Returns the loss and its gradients for W_r and b_r.
def balance_gradients(router_logits, router_inputs):
    pi = softmax(router_logits)  # (B, E)
    num_events, num_experts = pi.shape
    mean_pi = pi.mean(axis=0)

    loss = num_experts * np.sum((mean_pi - 1 / num_experts) ** 2)
    d_pi = np.broadcast_to(2 * num_experts * (mean_pi - 1 / num_experts) / num_events, pi.shape)
    d_z = pi * (d_pi - np.sum(d_pi * pi, axis=1, keepdims=True))  # through the softmax of each event

    return loss, {"W_r": d_z.T @ router_inputs, "b_r": d_z.sum(axis=0)}


# L2 penalty on the weight matrices (not the biases): 1/2 * sum of their squares, whose gradient is the weights
def l2_gradients(params):
    names = ("W_x", "W_h", "W_r", "W_y")
    loss = 0.5 * sum(float(np.sum(params[name] ** 2)) for name in names)
    return loss, {name: params[name] for name in names}


# Scales the gradients down so that their global norm (over all the arrays) is at most max_norm
def clip_global_norm(grads, max_norm):
    norm = np.sqrt(sum(float(np.sum(grad ** 2)) for grad in grads.values()))

    if norm > max_norm:
        scale = max_norm / norm
        for grad in grads.values():
            grad *= scale

    return norm
