import os
import tempfile
import unittest

import numpy as np

from ssnn.baselines import hidden_for_parameters, mlp_parameters
from ssnn.config import make_config
from ssnn.metrics import accuracy_by_group, routing_summary
from ssnn.gradients import add_task_gradients, task_gradients
from ssnn.model import PARAMETER_NAMES, SSNN, parameter_counts
from ssnn.reporting import append_row
from ssnn.tasks import Teachers, experiment_a, experiment_capacity, regime_examples
from ssnn.training import evaluate, train


class TasksTest(unittest.TestCase):
    def test_regime_examples(self):
        inputs, labels, regimes = regime_examples(1000, np.random.default_rng(0))

        self.assertEqual((inputs.shape, inputs.dtype), ((1000, 8), np.float32))
        np.testing.assert_array_equal(inputs[:, 4:].argmax(axis=1), regimes)  # the one-hot regime
        u = inputs[:, :4].astype(float)
        expected = [u[:, 0] * u[:, 1] > 0, u[:, 0] + u[:, 1] > 0, u[:, 2] * u[:, 3] > 0, np.abs(u[:, 0]) > np.abs(u[:, 1])]
        for regime in range(4):
            np.testing.assert_array_equal(labels[regimes == regime], expected[regime][regimes == regime])
        self.assertEqual(set(regimes), {0, 1, 2, 3})

    def test_splits_are_reproducible_and_independent(self):
        first, second = experiment_a(7, sizes=(50, 20, 20)), experiment_a(7, sizes=(80, 20, 20))

        np.testing.assert_array_equal(first[1][0], second[1][0])  # a larger training set doesn't change validation
        self.assertFalse(np.array_equal(first[1][0], first[2][0]))  # validation and test are different examples


class CapacityTaskTest(unittest.TestCase):
    def test_each_regime_follows_its_own_teacher(self):
        teachers = Teachers(num_regimes=5, seed=1)
        rng = np.random.default_rng(2)
        u = rng.normal(0, 1, (300, 8))

        for regime in range(5):
            expected = (np.tanh(u @ teachers.W1[regime].T) @ teachers.w2[regime] > 0).astype(int)
            np.testing.assert_array_equal(teachers.labels(u, np.full(300, regime)), expected)

        # Different regimes give different rules: the same inputs get other labels
        self.assertLess(np.mean(teachers.labels(u, np.zeros(300, int)) == teachers.labels(u, np.ones(300, int))), 0.9)

    def test_splits(self):
        (inputs, labels, regimes), val, test = experiment_capacity(3, num_regimes=16, per_regime=(50, 10, 10))

        self.assertEqual(inputs.shape, (800, 8 + 16))
        np.testing.assert_array_equal(inputs[:, 8:].argmax(axis=1), regimes)
        self.assertAlmostEqual(labels.mean(), 0.5, delta=0.06)  # odd teachers give balanced labels
        self.assertEqual(len(val[0]), 160)
        np.testing.assert_array_equal(experiment_capacity(3, 16, (50, 10, 10))[2][0], test[0])  # reproducible


class MetricsTest(unittest.TestCase):
    def test_routing_summary(self):
        # Group 0 always selects experts 0 and 1, group 1 experts 2 and 3: they share nothing (1 bit of 1 possible)
        selected = np.array([[0, 1], [0, 1], [2, 3], [2, 3]])
        summary = routing_summary(selected, num_experts=5, groups=[0, 0, 1, 1])

        np.testing.assert_allclose(summary["usage"], [0.25, 0.25, 0.25, 0.25, 0])
        self.assertEqual(summary["unused"], 1)
        self.assertAlmostEqual(summary["usage_entropy"], 2.0)
        self.assertAlmostEqual(summary["group_information"], 1.0)

        # Both groups select the same experts: the routing says nothing about the group
        same = routing_summary(np.array([[0, 1]] * 4), num_experts=2, groups=[0, 0, 1, 1])
        self.assertAlmostEqual(same["group_information"], 0.0)

    def test_accuracy_by_group(self):
        self.assertEqual(accuracy_by_group([1, 0, 1, 1], [1, 1, 1, 0], [0, 0, 1, 1]), {0: 0.5, 1: 0.5})


class TrainingTest(unittest.TestCase):
    def small_training(self, seed=0, **changes):
        config = make_config(seed=seed, num_experts=4, state_dim=8, **changes)
        data = regime_examples(32, np.random.default_rng(seed))
        model = SSNN(config)
        history = train(model, data, data)
        return model, history, data

    def test_can_overfit_32_examples(self):
        # A small batch that the model can learn by heart: if it can't, something is broken
        model, history, data = self.small_training(epochs=150, learning_rate=0.01)
        probabilities, _ = evaluate(model, data[0])

        self.assertLess(history["loss"][-1], history["loss"][0] / 3)
        self.assertGreaterEqual(np.mean(probabilities.argmax(axis=1) == data[1]), 0.9)

    def test_same_seed_same_training(self):
        first, _, _ = self.small_training(epochs=3)
        second, _, _ = self.small_training(epochs=3)
        for name in first.params:
            np.testing.assert_array_equal(first.params[name], second.params[name])

    def test_no_nan_or_inf_after_short_trainings(self):
        for seed in range(3):
            for changes in ({}, {"router": "random"}, {"top_k": 4}, {"l2_weight": 1e-3}):
                with self.subTest(seed=seed, **changes):
                    model, history, _ = self.small_training(seed=seed, epochs=3, **changes)
                    self.assertTrue(all(np.all(np.isfinite(array)) for array in model.params.values()))
                    self.assertTrue(np.all(np.isfinite(history["loss"])))

    def test_keeps_the_epoch_with_the_lowest_validation_loss(self):
        _, history, _ = self.small_training(epochs=20, learning_rate=0.05)
        self.assertEqual(history["best_epoch"], int(np.argmin(history["val_loss"])) + 1)


class GradientSumTest(unittest.TestCase):
    def test_adding_in_place_gives_the_sum_of_the_gradients(self):
        model = SSNN(make_config(num_experts=6, state_dim=4))
        rng = np.random.default_rng(3)
        total = {name: np.zeros_like(model.params[name]) for name in PARAMETER_NAMES}
        expected = {name: np.zeros_like(model.params[name]) for name in PARAMETER_NAMES}

        for label in (0, 1, 1):
            _, _, trace = model.step(rng.normal(0, 1, 8), model.new_state())
            add_task_gradients(model, trace, label, total)
            for name, grad in task_gradients(model, trace, label)[1].items():
                expected[name] += grad

        for name in PARAMETER_NAMES:
            np.testing.assert_allclose(total[name], expected[name], rtol=1e-6, atol=1e-8)


class ReportTest(unittest.TestCase):
    def test_a_file_with_other_columns_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "report.csv")
            append_row(filepath, {"a": 1, "b": 2})
            append_row(filepath, {"a": 3, "b": 4})
            with self.assertRaises(ValueError):
                append_row(filepath, {"a": 5, "c": 6})

            with open(filepath) as file:
                self.assertEqual(file.read().splitlines(), ["a,b", "1,2", "3,4"])


class BaselinesTest(unittest.TestCase):
    def test_mlp_sizes_match_the_ssnn(self):
        counts = parameter_counts(make_config())
        for target in counts.values():
            hidden = hidden_for_parameters(target, 8, 2)
            self.assertLess(abs(mlp_parameters(8, hidden, 2) - target), 11)  # within one hidden neuron


if __name__ == "__main__":
    unittest.main()
