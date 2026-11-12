import unittest

import numpy as np

from ssnn.config import make_config
from ssnn.inference import AllExpertsEngine, DenseRNN, SelectiveEngine, rnn_hidden_for
from ssnn.model import SSNN, State


def copy_state(state):
    return State(state.h.copy(), state.g.copy())


class EnginesTest(unittest.TestCase):
    # A session of 20 events: the lean engines must follow model.step, event after event
    def assert_same_session(self, model, engine):
        rng = np.random.default_rng(5)
        reference, state = model.new_state(), model.new_state()

        for _ in range(20):
            x = rng.normal(0, 1, model.config["input_dim"]).astype(model.dtype)
            expected, reference, _ = model.step(x, reference)
            probabilities = engine.step(x, state)

            np.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(state.h, reference.h, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(state.g, reference.g, rtol=1e-5, atol=1e-6)

    def test_selective_engine_matches_the_model(self):
        for changes in ({}, {"top_k": 1}, {"top_k": 5, "routing_temperature": 0.5}, {"top_k": 16},
                        {"num_experts": 1100, "state_dim": 4, "top_k": 3}):  # over 1024 experts: np.argpartition
            with self.subTest(**changes):
                model = SSNN(make_config(input_dim=6, **changes))
                self.assert_same_session(model, SelectiveEngine(model))

    def test_all_experts_engine_matches_the_model_with_k_equal_to_e(self):
        model = SSNN(make_config(input_dim=6, top_k=16))
        self.assert_same_session(model, AllExpertsEngine(model))

    def test_profile_marks_the_five_parts(self):
        model = SSNN(make_config())
        marks = []
        SelectiveEngine(model).step(np.ones(8, np.float32), model.new_state(), marks)
        self.assertEqual(len(marks), 5)
        self.assertEqual(marks, sorted(marks))

    def test_numbers_touched_per_event(self):
        model = SSNN(make_config())  # E = 16, H = 32, d = 8, k = 2
        self.assertEqual(SelectiveEngine(model).touched, 2 * (32 * 8 + 32 * 32 + 32) + 16 * 41 + 2 * 41)
        self.assertEqual(AllExpertsEngine(model).touched, sum(array.size for array in model.params.values()))

    def test_dense_rnn(self):
        rnn = DenseRNN(input_dim=8, hidden_dim=20, num_classes=2)
        state = rnn.new_state()
        first = rnn.step(np.ones(8, np.float32), state).copy()
        second = rnn.step(np.ones(8, np.float32), state)

        self.assertAlmostEqual(float(first.sum()), 1.0, places=6)
        self.assertFalse(np.allclose(first, second))  # the hidden state carried over
        self.assertEqual(rnn.touched, 20 * 28 + 20 + 2 * 28 + 2)  # 28 = 8 inputs + 20 hidden values

    def test_rnn_hidden_size_for_a_budget(self):
        for target in (3362, 147968, 9_000_000):
            hidden = rnn_hidden_for(target, 32, 2)
            touched = DenseRNN(32, hidden, 2).touched
            self.assertLess(abs(touched - target) / target, 0.02)


if __name__ == "__main__":
    unittest.main()
