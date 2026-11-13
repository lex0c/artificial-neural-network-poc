import unittest

import numpy as np

from activations import softmax
from gradcheck import numerical_gradients
from ssnn.config import make_config
from ssnn.gradients import task_gradients
from ssnn.model import SSNN, State
from ssnn.recurrent import GRU, SequenceSSNN, TanhRNN, gru_hidden_for
from ssnn.metrics import memory_leak
from ssnn.tasks import MemoryTask, experiment_memory
from ssnn.training import evaluate_sequences, train_sequences

# One register and no noise: a single bit, inputs SET(0, 0), SET(0, 1), QUERY(0) and NOOP. Its oracle routing gives
# the writes and the QUERY to the memory expert 0, and the NOOPs to expert 1.
BIT = MemoryTask(registers=1, noise=0)
ORACLE = BIT.oracle_routes()


def bit_sequence(*events):
    rng = np.random.default_rng(0)
    vectors = {"0": BIT.set_event(0, 0), "1": BIT.set_event(0, 1), "?": BIT.query_event(0)}
    return np.array([vectors[event] if event in vectors else BIT.noop_event(rng) for event in events], dtype=float)


class MemoryTaskTest(unittest.TestCase):
    def test_the_answer_is_the_last_write_of_the_queried_register(self):
        task = MemoryTask(registers=4, noise=3, rewrite_rate=0.3)
        rng = np.random.default_rng(0)

        for _ in range(200):
            inputs, label, distance = task.sequence(rng, int(rng.integers(1, 40)))
            register = int(inputs[-1, 8:12].argmax())  # the QUERY, after the 8 SET inputs
            writes = [t for t in range(len(inputs)) if inputs[t, 2 * register:2 * register + 2].any()]

            self.assertEqual(inputs[-1, 8 + register], 1)
            self.assertEqual(label, int(inputs[writes[-1], 2 * register + 1]))  # SET(i, 1) or SET(i, 0)
            self.assertEqual(len(inputs) - 1 - writes[-1], distance)

    def test_registers_of_several_values(self):
        task = MemoryTask(registers=3, noise=2, rewrite_rate=0.3, values=4)
        rng = np.random.default_rng(3)
        self.assertEqual(task.input_dim, 4 * 3 + 3 + 1 + 2)

        for _ in range(100):
            inputs, label, distance = task.sequence(rng, int(rng.integers(1, 20)))
            register = int(inputs[-1, 12:15].argmax())  # the QUERY, after the 12 SET inputs
            writes = [t for t in range(len(inputs)) if inputs[t, 4 * register:4 * register + 4].any()]
            self.assertEqual(label, int(inputs[writes[-1], 4 * register:4 * register + 4].argmax()))
            self.assertEqual(len(inputs) - 1 - writes[-1], distance)

        routes = task.oracle_routes()
        for register in range(3):
            for value in range(4):
                self.assertEqual(routes[int(task.set_event(register, value).argmax())], [register])

    def test_keyed_registers(self):
        # [register (3), value (4), SET, QUERY, NOOP, noise (2)]
        task = MemoryTask(registers=3, noise=2, rewrite_rate=0.3, values=4, encoding="keyed")
        rng = np.random.default_rng(5)
        self.assertEqual((task.input_dim, task.noop_index), (3 + 4 + 3 + 2, 9))
        np.testing.assert_array_equal(task.set_event(2, 1)[:9], [0, 0, 1, 0, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(task.query_event(1)[:9], [0, 1, 0, 0, 0, 0, 0, 0, 1])

        for _ in range(100):
            inputs, label, distance = task.sequence(rng, int(rng.integers(1, 20)))
            register = int(inputs[-1, :3].argmax())
            self.assertEqual(inputs[-1, 8], 1)  # a QUERY
            writes = [t for t in range(len(inputs)) if inputs[t, register] == 1 and inputs[t, 7] == 1]
            self.assertEqual(label, int(inputs[writes[-1], 3:7].argmax()))
            self.assertEqual(len(inputs) - 1 - writes[-1], distance)

        routes = task.oracle_routes()
        self.assertEqual(len(routes), task.input_dim)
        for register in range(3):
            for event in [task.set_event(register, value) for value in range(4)] + [task.query_event(register)]:
                self.assertEqual(routes[int(event.argmax())], [register])
        self.assertEqual(routes[int(task.noop_event(rng).argmax())], [3])

        with self.assertRaises(ValueError):
            MemoryTask(encoding="binary")

    def test_only_the_query_tells_which_value_to_answer(self):
        # Over sequences of 8 registers of one bit: how many positions the last write of the queried register takes,
        # how often it replaces another value, and how often a rule that ignores the QUERY (the value of the last write
        # before the first NOOP) is right
        def measure(task):
            rng = np.random.default_rng(8)
            positions, replaced, rule = set(), [], []
            for _ in range(500):
                inputs, label, distance = task.sequence(rng, int(rng.integers(1, 101)))
                register, last = int(inputs[-1].argmax()) - 16, len(inputs) - 1 - distance
                positions.add(last)
                replaced.append(any(inputs[t, 2 * register + 1 - label] == 1 for t in range(last)))
                noops = np.flatnonzero(inputs[:, task.noop_index])
                written = [int(x.argmax()) % 2 for x in inputs[:noops[0] if len(noops) else -1] if x.argmax() < 16]
                rule.append(written[-1] == label)
            return len(positions), np.mean(replaced), np.mean(rule)

        positions, replaced, rule = measure(MemoryTask(registers=8))
        self.assertGreater(positions, 25)
        self.assertGreater(replaced, 0.4)
        self.assertLess(rule, 0.6)  # a guess is right half the time

        # The first version: the last write always right after the first writes, repeating the value of the first one
        positions, replaced, rule = measure(MemoryTask(registers=8, preamble=0, new_value=False))
        self.assertEqual((positions, replaced), (1, 0.0))
        self.assertGreater(rule, 0.85)

        with self.assertRaises(ValueError):
            MemoryTask(preamble=-1)

    def test_noise_only_on_the_noops(self):
        task = MemoryTask(registers=2, noise=4)
        inputs, _, _ = task.sequence(np.random.default_rng(1), 20)
        noops = inputs[:, task.noop_index] == 1

        self.assertTrue(np.all(inputs[~noops, task.noop_index + 1:] == 0))
        self.assertTrue(np.all(inputs[noops, task.noop_index + 1:] != 0))
        self.assertEqual(task.input_dim, 3 * 2 + 1 + 4)

    def test_oracle_routes(self):
        task = MemoryTask(registers=3, noise=2)
        routes = task.oracle_routes()
        self.assertEqual(len(routes), task.input_dim)
        for register in range(3):
            for event in (task.set_event(register, 0), task.set_event(register, 1), task.query_event(register)):
                self.assertEqual(routes[int(event.argmax())], [register])
        self.assertEqual(routes[int(task.noop_event(np.random.default_rng(2)).argmax())], [3])

    def test_splits(self):
        train, val, tests = experiment_memory(1, MemoryTask(registers=2, noise=2), sizes=(50, 10), test_size=10)
        self.assertEqual((len(train), len(val)), (50, 10))
        for (low, high), examples in tests.items():
            self.assertTrue(all(low <= distance <= high for _, _, distance in examples))


class SequenceGradientTest(unittest.TestCase):
    # A sequence with two writes and gaps, checked in float64 against central differences
    SEQUENCE = bit_sequence("1", ".", ".", "0", ".", ".", ".", "?")

    def check(self, model, label=0, before_each_forward=None):
        def loss():
            if before_each_forward:
                before_each_forward()
            probabilities, _ = model.forward(self.SEQUENCE)
            return -np.log(probabilities[label])

        if before_each_forward:
            before_each_forward()
        _, cache = model.forward(self.SEQUENCE)
        value, grads = model.gradients(cache, label)
        self.assertAlmostEqual(value, loss())

        expected = numerical_gradients(loss, [model.params[name] for name in model.names])
        for name, numerical in zip(model.names, expected):
            np.testing.assert_allclose(grads[name], numerical, rtol=1e-5, atol=1e-8, err_msg=name)
        return grads

    def ssnn(self, **changes):
        config = make_config(dtype="float64", input_dim=4, state_dim=5, num_classes=2, **changes)
        return SequenceSSNN(SSNN(config, seed=3))

    def test_ssnn_with_fixed_routing(self):
        self.check(self.ssnn(num_experts=2, top_k=1, router="fixed", fixed_routes=ORACLE))

    def test_ssnn_with_a_learned_router_over_all_experts(self):
        # k = E: the selection never changes, so the router's gradient through g_(t-1) is checked over the whole sequence
        self.check(self.ssnn(num_experts=3, top_k=3, routing_temperature=0.7))

    def test_ssnn_with_a_learned_top_k(self):
        model = self.ssnn(num_experts=4, top_k=2)
        _, traces = model.forward(self.SEQUENCE)
        for trace in traces:  # the nudges of the check can't change any selection
            ordered = np.sort(trace["router_logits"])[::-1]
            self.assertGreater(ordered[1] - ordered[2], 1e-4)
        self.check(model)

    def test_ssnn_with_a_learned_top_k_and_the_gated_write(self):
        model = self.ssnn(num_experts=4, top_k=2, write="gated")
        _, traces = model.forward(self.SEQUENCE)
        for trace in traces:
            ordered = np.sort(trace["router_logits"])[::-1]
            self.assertGreater(ordered[1] - ordered[2], 1e-4)
        self.check(model)

    def test_ssnn_with_a_random_router(self):
        model = self.ssnn(num_experts=4, top_k=2, router="random")

        def same_draws():
            model.model.routing_rng = np.random.default_rng(8)

        grads = self.check(model, before_each_forward=same_draws)
        self.assertFalse(np.any(grads["W_r"]))

    def test_dense_rnn(self):
        self.check(TanhRNN(input_dim=4, hidden_dim=5, num_classes=2, seed=1, dtype="float64"), label=1)

    def test_gru(self):
        self.check(GRU(input_dim=4, hidden_dim=5, num_classes=2, seed=1, dtype="float64"), label=1)

    def test_gru_that_keeps_its_state_by_default(self):
        model = GRU(input_dim=4, hidden_dim=5, num_classes=2, seed=1, dtype="float64", update_bias=2.0)
        self.assertTrue(np.all(model.params["b_z"] == 2) and not np.any(model.params["b_r"]))
        self.check(model, label=1)


# The loss of a sequence if expert were added to the selection of event step with the given weight, the weights of the
# selected experts scaled by 1 - weight: a forward pass of its own (gated write), from the given state or the zero one
def loss_with_an_added_expert(ssnn, inputs, label, step, expert, weight, state=None):
    p, rho, beta = ssnn.params, ssnn.config["rho"], ssnn.config["beta"]
    state = state or ssnn.new_state()

    for t, x in enumerate(inputs):
        _, selected, weights = ssnn.route(np.concatenate([x, state.g]))
        if t == step:
            selected, weights = np.append(selected, expert), np.append(weights * (1 - weight), weight)
        h = state.h.copy()
        for i, a in zip(selected, weights):
            n = np.tanh(p["W_x"][i] @ x + p["W_h"][i] @ state.h[i] + p["b"][i])
            h[i] = (1 - rho * a) * state.h[i] + rho * a * n
        state = State(h, (1 - beta) * state.g + beta * (weights @ h[selected]))

    return -np.log(softmax(p["W_y"] @ np.concatenate([x, state.g]) + p["b_y"])[label])


class DenseRouterGradientTest(unittest.TestCase):
    TAU = 0.7

    def models(self):
        # The same parameters, with the router gradient of the selected experts only and with the dense one
        return [SSNN(make_config(dtype="float64", input_dim=4, state_dim=5, num_classes=2, num_experts=5, top_k=2,
                                 write="gated", routing_temperature=self.TAU, router_gradient=mode), seed=3)
                for mode in ("selected", "dense")]

    # What the dense gradient adds to b_r and W_r: for each event and each expert left out, a_e (1 - a_e) c_e / tau,
    # with c_e, the derivative of the loss with respect to the weight of the added expert, by central differences
    def expected_additions(self, model, traces, loss_with, eps=1e-6):
        b_r, W_r = np.zeros_like(model.params["b_r"]), np.zeros_like(model.params["W_r"])
        for t, trace in enumerate(traces):
            ordered = np.sort(trace["router_logits"])[::-1]
            self.assertGreater(ordered[1] - ordered[2], 1e-4)  # the nudges can't change a selection
            z = trace["router_logits"] / self.TAU
            for e in sorted(set(range(len(z))) - set(trace["selected"].tolist())):
                c = (loss_with(t, e, eps) - loss_with(t, e, -eps)) / (2 * eps)
                a = np.exp(z[e]) / (np.exp(z[e]) + np.exp(z[trace["selected"]]).sum())
                b_r[e] += a * (1 - a) * c / self.TAU
                W_r[e] += a * (1 - a) * c / self.TAU * trace["q"]
        return b_r, W_r

    def check(self, grads, expected):
        for name in ("W_x", "W_h", "b", "W_y", "b_y"):  # exact, unchanged
            np.testing.assert_array_equal(grads["dense"][name], grads["selected"][name], err_msg=name)
        for name, added in zip(("b_r", "W_r"), expected):
            self.assertGreater(np.abs(added).max(), 1e-6)
            np.testing.assert_allclose(grads["dense"][name] - grads["selected"][name], added, rtol=1e-5, atol=1e-10,
                                       err_msg=name)

    def test_over_a_sequence(self):
        sequence, label = SequenceGradientTest.SEQUENCE, 1
        grads = {}
        for model in self.models():
            sequence_model = SequenceSSNN(model)
            _, traces = sequence_model.forward(sequence)
            grads[model.config["router_gradient"]] = sequence_model.gradients(traces, label)[1]

        self.check(grads, self.expected_additions(model, traces, lambda t, e, weight: loss_with_an_added_expert(
            model, sequence, label, t, e, weight)))

    def test_on_one_event_from_a_given_state(self):
        # Independent examples treat the state before the event as a constant: only the read of the added expert counts
        rng = np.random.default_rng(4)
        x, label = rng.normal(0, 1, 4), 0
        state = State(rng.normal(0, 0.5, (5, 5)), rng.normal(0, 0.5, 5))
        grads = {}
        for model in self.models():
            _, _, trace = model.step(x, state)
            grads[model.config["router_gradient"]] = task_gradients(model, trace, label)[1]

        self.check(grads, self.expected_additions(model, [trace], lambda t, e, weight: loss_with_an_added_expert(
            model, [x], label, t, e, weight, state)))

    def test_one_expert_per_event_gets_a_router_gradient(self):
        # With k = 1 the weight of the only expert is always 1: without the dense gradient, the task gives the router
        # nothing to learn from
        trained = {}
        for mode in ("selected", "dense"):
            model = SequenceSSNN(SSNN(make_config(dtype="float64", input_dim=4, state_dim=5, num_experts=3, top_k=1,
                                                  write="gated", router_gradient=mode), seed=1))
            _, traces = model.forward(SequenceGradientTest.SEQUENCE)
            trained[mode] = model.gradients(traces, 1)[1]["W_r"]
        self.assertFalse(np.any(trained["selected"]))
        self.assertTrue(np.any(trained["dense"]))


class MemoryGradientTest(unittest.TestCase):
    def test_the_gradient_of_the_memory_expert_doesnt_fade_with_the_wait(self):
        # With the oracle, the memory expert waits through the NOOPs unchanged: the gradient that reaches its input
        # weights from the QUERY is the same after 50 or 200 NOOPs
        config = make_config(dtype="float64", input_dim=4, state_dim=8, num_experts=2, top_k=1, router="fixed", fixed_routes=ORACLE)
        model = SequenceSSNN(SSNN(config, seed=2))

        def memory_gradient(gap):
            _, cache = model.forward(bit_sequence("1", *["."] * gap, "?"))
            return model.gradients(cache, label=1)[1]["W_x"][0]

        short, long = memory_gradient(50), memory_gradient(200)
        self.assertGreater(np.abs(short).max(), 1e-3)
        np.testing.assert_allclose(long, short, rtol=1e-6, atol=1e-12)


class MemoryTrainingTest(unittest.TestCase):
    def test_oracle_ssnn_learns_to_remember(self):
        rng = np.random.default_rng(4)
        train, val = BIT.examples(300, (1, 10), rng), BIT.examples(100, (1, 10), rng)
        config = make_config(input_dim=4, state_dim=8, num_experts=2, top_k=1, router="fixed", fixed_routes=ORACLE)
        model = SequenceSSNN(SSNN(config, seed=0))

        train_sequences(model, train, val, epochs=15, learning_rate=0.01)
        self.assertGreater(evaluate_sequences(model, val)[1], 0.95)

    def test_balance_loss_reaches_the_router(self):
        rng = np.random.default_rng(5)
        examples = BIT.examples(8, (1, 4), rng)
        trained = {}
        for weight in (0.0, 1.0):
            model = SequenceSSNN(SSNN(make_config(input_dim=4, state_dim=4, num_experts=4, top_k=2), seed=1))
            train_sequences(model, examples, examples, epochs=1, learning_rate=0.01, balance_weight=weight)
            trained[weight] = model.params["W_r"].copy()
        self.assertFalse(np.allclose(trained[0.0], trained[1.0]))

    def test_memory_leak(self):
        # A write, three NOOPs and a QUERY that reads expert 0: the second NOOP selected it with weight 0.25, the others
        # didn't. With the gated write it wrote rho * 0.25 to it, with the fixed one rho.
        inputs = bit_sequence("1", ".", ".", ".", "?")
        traces = [{"selected": [0], "weights": [1.0]}, {"selected": [1], "weights": [1.0]},
                  {"selected": [1, 0], "weights": [0.75, 0.25]}, {"selected": [2], "weights": [1.0]},
                  {"selected": [0, 3], "weights": [0.5, 0.5]}]
        for write, strength in (("fixed", 0.35), ("gated", 0.35 * 0.25)):
            leak = memory_leak([traces], [inputs], [4], BIT.noop_index, make_config(write=write))
            self.assertAlmostEqual(leak["touched"], 1 / 3)
            self.assertAlmostEqual(leak["write"], strength / 3)

        # The oracle never touches the memory in the gaps
        model = SequenceSSNN(SSNN(make_config(input_dim=4, state_dim=4, num_experts=2, top_k=1, router="fixed", fixed_routes=ORACLE)))
        examples = BIT.examples(5, (2, 9), np.random.default_rng(6))
        traces = [model.forward(inputs)[1] for inputs, _, _ in examples]
        leak = memory_leak(traces, [e[0] for e in examples], [e[2] for e in examples], BIT.noop_index, model.model.config)
        self.assertEqual(leak, {"touched": 0.0, "write": 0.0})

    def test_gru_size_for_a_budget(self):
        for target in (1258, 5000):
            touched = GRU(4, gru_hidden_for(target, 4, 2), 2).touched
            self.assertLess(abs(touched - target) / target, 0.08)


if __name__ == "__main__":
    unittest.main()
