import unittest

import numpy as np

from gradcheck import numerical_gradients
from ssnn.config import make_config
from ssnn.model import SSNN
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
        # A write, three NOOPs and a QUERY that reads expert 0: the second NOOP selected it, the others didn't
        inputs = bit_sequence("1", ".", ".", ".", "?")
        traces = [{"selected": [0]}, {"selected": [1]}, {"selected": [1, 0]}, {"selected": [2]}, {"selected": [0, 3]}]
        self.assertAlmostEqual(memory_leak([traces], [inputs], [4], BIT.noop_index), 1 / 3)

        # The oracle never touches the memory in the gaps
        model = SequenceSSNN(SSNN(make_config(input_dim=4, state_dim=4, num_experts=2, top_k=1, router="fixed", fixed_routes=ORACLE)))
        examples = BIT.examples(5, (2, 9), np.random.default_rng(6))
        traces = [model.forward(inputs)[1] for inputs, _, _ in examples]
        self.assertEqual(memory_leak(traces, [e[0] for e in examples], [e[2] for e in examples], BIT.noop_index), 0.0)

    def test_gru_size_for_a_budget(self):
        for target in (1258, 5000):
            touched = GRU(4, gru_hidden_for(target, 4, 2), 2).touched
            self.assertLess(abs(touched - target) / target, 0.08)


if __name__ == "__main__":
    unittest.main()
