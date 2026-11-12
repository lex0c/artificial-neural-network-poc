import os
import tempfile
import unittest

import numpy as np

from activations import softmax
from gradcheck import numerical_gradients
from ssnn.config import DEFAULT_CONFIG, config_hash, make_config
from ssnn.gradients import balance_gradients, task_gradients
from ssnn.model import PARAMETER_NAMES, SSNN, State, parameter_counts


def random_state(model, rng):
    # A state that isn't zero, so the recurrent weights W_h matter
    E, H = model.config["num_experts"], model.config["state_dim"]
    return State(rng.normal(0, 0.5, (E, H)).astype(model.dtype), rng.normal(0, 0.5, H).astype(model.dtype))


# The same event computed by running all the experts and masking those outside the selection: the selective
# execution must give the same result, so a gain can't come from a bug that skips work
def dense_masked_step(model, x, state):
    p, rho, beta = model.params, model.config["rho"], model.config["beta"]
    logits, selected, weights = model.route(np.concatenate([x, state.g]))
    mask = np.zeros(model.config["num_experts"], bool)
    mask[selected] = True
    all_weights = np.zeros(model.config["num_experts"], model.dtype)
    all_weights[selected] = weights

    n = np.tanh(np.einsum("ehd,d->eh", p["W_x"], x) + np.einsum("ehk,ek->eh", p["W_h"], state.h) + p["b"])
    h = np.where(mask[:, None], (1 - rho) * state.h + rho * n, state.h)
    g = (1 - beta) * state.g + beta * (all_weights @ h)
    return softmax(p["W_y"] @ np.concatenate([x, g]) + p["b_y"]), h, g


class ForwardTest(unittest.TestCase):
    def setUp(self):
        self.model = SSNN(DEFAULT_CONFIG)
        self.rng = np.random.default_rng(1)
        self.x = self.rng.normal(0, 1, DEFAULT_CONFIG["input_dim"]).astype(np.float32)

    def test_shapes_and_float32(self):
        probabilities, state, trace = self.model.step(self.x, self.model.new_state())

        self.assertEqual(probabilities.shape, (DEFAULT_CONFIG["num_classes"],))
        self.assertEqual(state.h.shape, (DEFAULT_CONFIG["num_experts"], DEFAULT_CONFIG["state_dim"]))
        self.assertEqual(state.g.shape, (DEFAULT_CONFIG["state_dim"],))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        for name, array in [("probabilities", probabilities), ("h", state.h), ("g", state.g)] + list(self.model.params.items()):
            self.assertEqual(array.dtype, np.float32, name)
        self.assertEqual(len(trace["selected"]), DEFAULT_CONFIG["top_k"])

    def test_inactive_experts_keep_their_state_exactly(self):
        state = random_state(self.model, self.rng)
        _, new_state, trace = self.model.step(self.x, state)
        inactive = np.setdiff1d(np.arange(DEFAULT_CONFIG["num_experts"]), trace["selected"])

        np.testing.assert_array_equal(new_state.h[inactive], state.h[inactive])
        self.assertFalse(np.array_equal(new_state.h[trace["selected"]], state.h[trace["selected"]]))

    def test_step_doesnt_change_the_given_state(self):
        state = random_state(self.model, self.rng)
        h, g = state.h.copy(), state.g.copy()
        self.model.step(self.x, state)

        np.testing.assert_array_equal(state.h, h)
        np.testing.assert_array_equal(state.g, g)

    def test_sessions_are_independent(self):
        # Two sessions fed different events don't affect each other, and a new state starts at zero
        a, b = self.model.new_state(), self.model.new_state()
        for _ in range(5):
            _, a, _ = self.model.step(self.rng.normal(0, 1, DEFAULT_CONFIG["input_dim"]), a)

        _, b_after, _ = self.model.step(self.x, b)
        _, fresh_after, _ = self.model.step(self.x, self.model.new_state())
        np.testing.assert_array_equal(b_after.h, fresh_after.h)
        self.assertFalse(np.any(self.model.new_state().h) or np.any(self.model.new_state().g))

    def test_selective_execution_matches_the_dense_masked_one(self):
        state = random_state(self.model, self.rng)
        probabilities, new_state, _ = self.model.step(self.x, state)
        dense_probabilities, h, g = dense_masked_step(self.model, self.x, state)

        np.testing.assert_allclose(probabilities, dense_probabilities, rtol=1e-5)
        np.testing.assert_allclose(new_state.h, h, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(new_state.g, g, rtol=1e-5, atol=1e-7)

    def test_top_k_is_reproducible_and_breaks_ties_by_index(self):
        self.model.params["W_r"][:] = 0  # every logit equal
        _, selected, weights = self.model.route(np.zeros(DEFAULT_CONFIG["input_dim"] + DEFAULT_CONFIG["state_dim"], np.float32))

        self.assertEqual(list(selected), [0, 1])
        np.testing.assert_allclose(weights, [0.5, 0.5])

    def test_extreme_numbers_of_experts(self):
        for changes in ({"top_k": 1}, {"top_k": 16}, {"num_experts": 1, "top_k": 1}):
            with self.subTest(**changes):
                model = SSNN(make_config(**changes))
                probabilities, state, trace = model.step(self.x, model.new_state())
                self.assertTrue(np.all(np.isfinite(probabilities)) and np.all(np.isfinite(state.h)))

    def test_softmax_is_stable_with_large_logits(self):
        np.testing.assert_allclose(softmax(np.array([1000.0, -1000.0, 999.0], np.float32)), [0.7311, 0.0, 0.2689], atol=1e-4)

    def test_random_router_draws_k_experts_with_equal_weights(self):
        model = SSNN(make_config(router="random"))
        _, selected, weights = model.route(np.zeros(DEFAULT_CONFIG["input_dim"] + DEFAULT_CONFIG["state_dim"], np.float32))

        self.assertEqual(len(set(selected)), DEFAULT_CONFIG["top_k"])
        np.testing.assert_allclose(weights, 0.5)

    def test_saved_model_repeats_its_inferences(self):
        state = random_state(self.model, self.rng)
        probabilities, _, _ = self.model.step(self.x, state)

        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "ssnn.npz")
            self.model.save(filepath)
            loaded = SSNN.load(filepath)

        np.testing.assert_array_equal(loaded.step(self.x, state)[0], probabilities)
        self.assertEqual(loaded.config, self.model.config)

    def test_parameter_counts(self):
        # The small configuration (about 82 KiB of expert weights in float32) and a large one (73,984 parameters per expert)
        small = parameter_counts(DEFAULT_CONFIG)
        large = parameter_counts(make_config(input_dim=32, state_dim=256, num_experts=128))
        self.assertEqual(sum(self.model.params[name].size for name in PARAMETER_NAMES), small["total"])
        self.assertAlmostEqual(16 * (32 * 8 + 32 * 32 + 32) * 4 / 1024, 82, delta=1)
        self.assertEqual(large["per_event"] - 128 * (32 + 256 + 1) - 2 * (32 + 256 + 1), 2 * 73984)

    def test_config_is_checked_and_fingerprinted(self):
        with self.assertRaises(ValueError):
            make_config(top_k=17)
        with self.assertRaises(ValueError):
            make_config(learning_rat=0.1)  # a typo
        with self.assertRaises(ValueError):
            make_config(router_gradient="dense")  # needs the gated write
        with self.assertRaises(ValueError):
            make_config(router_gradient="dense", write="gated", router="random")
        self.assertEqual(config_hash(make_config()), config_hash(dict(DEFAULT_CONFIG)))
        self.assertNotEqual(config_hash(make_config(top_k=3)), config_hash(DEFAULT_CONFIG))


class GradientTest(unittest.TestCase):
    def check(self, config, label=1):
        # In float64: central differences in float32 are too coarse. The state isn't zero, so W_h gets a gradient
        model = SSNN(make_config(dtype="float64", **config))
        rng = np.random.default_rng(2)
        state = random_state(model, rng)
        x = rng.normal(0, 1, model.config["input_dim"])

        def step():
            model.routing_rng = np.random.default_rng(4)  # the random router draws the same experts every time
            return model.step(x, state)

        _, _, trace = step()
        if trace["router_logits"] is not None:
            ordered = np.sort(trace["router_logits"])[::-1]
            k = model.config["top_k"]
            if k < model.config["num_experts"]:
                self.assertGreater(ordered[k - 1] - ordered[k], 1e-3)  # the top-k choice can't change with the nudges

        loss, grads = task_gradients(model, trace, label)
        self.assertAlmostEqual(loss, -np.log(trace["probabilities"][label]))

        expected = numerical_gradients(lambda: -np.log(step()[0][label]), model.parameters())
        for name, numerical in zip(PARAMETER_NAMES, expected):
            np.testing.assert_allclose(grads[name], numerical, rtol=1e-5, atol=1e-8, err_msg=name)
        return grads

    def test_one_event_matches_numerical_gradients(self):
        self.check({"num_experts": 5, "top_k": 2, "state_dim": 4, "input_dim": 3, "num_classes": 3})

    def test_with_the_gated_write(self):
        # The weights of the router also set how much each expert writes, so they get a gradient through the write too
        self.check({"num_experts": 5, "top_k": 3, "state_dim": 4, "input_dim": 3, "write": "gated"})

    def test_with_a_temperature_and_all_experts(self):
        self.check({"num_experts": 4, "top_k": 4, "state_dim": 3, "input_dim": 2, "routing_temperature": 0.5})

    def test_random_router_gets_no_gradient(self):
        grads = self.check({"num_experts": 4, "top_k": 2, "state_dim": 3, "input_dim": 2, "router": "random"})
        self.assertFalse(np.any(grads["W_r"]) or np.any(grads["b_r"]))

    def test_router_gets_no_task_gradient_with_k_1(self):
        # The softmax of a single selected expert is always 1: the reason the default is k = 2
        grads = self.check({"num_experts": 4, "top_k": 1, "state_dim": 3, "input_dim": 2})
        self.assertFalse(np.any(grads["W_r"]) or np.any(grads["b_r"]))

    def test_balance_loss_matches_numerical_gradients(self):
        rng = np.random.default_rng(3)
        W_r, b_r, inputs = rng.normal(0, 1, (5, 4)), rng.normal(0, 1, 5), rng.normal(0, 1, (6, 4))

        def loss():
            return balance_gradients(inputs @ W_r.T + b_r, inputs)[0]

        _, grads = balance_gradients(inputs @ W_r.T + b_r, inputs)
        numerical_W, numerical_b = numerical_gradients(loss, [W_r, b_r])
        np.testing.assert_allclose(grads["W_r"], numerical_W, rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(grads["b_r"], numerical_b, rtol=1e-5, atol=1e-9)

    def test_balance_loss_is_zero_for_a_uniform_router(self):
        loss, _ = balance_gradients(np.zeros((4, 8)), np.ones((4, 3)))
        self.assertAlmostEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
