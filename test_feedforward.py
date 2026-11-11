import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
from joblib import load

from feedforward import FeedForward, load_model, clone_model
from etc import (
    loss_fn, loss_derivative_fn, softmax_crossentropy_delta, to_categorical, set_seed,
    softmax, sigmoid, sigmoid_derivative, relu, relu_derivative,
)


NUM_SAMPLES = 5
NUM_INPUTS = 3
NUM_HIDDEN = 4
NUM_OUTPUTS = 3


def build_model(hidden_act, output_act, loss):
    model = FeedForward()
    model.add_layer(num_inputs=NUM_INPUTS, num_neurons=NUM_HIDDEN, act_fn=hidden_act)
    model.add_layer(num_inputs=NUM_HIDDEN, num_neurons=NUM_OUTPUTS, act_fn=output_act)
    model.configure(loss=loss)
    return model


def make_targets(loss, output_act, rng):
    if loss == "sparse_categorical_crossentropy":
        return rng.integers(0, NUM_OUTPUTS, NUM_SAMPLES)
    if loss == "categorical_crossentropy":
        return to_categorical(rng.integers(0, NUM_OUTPUTS, NUM_SAMPLES), num_classes=NUM_OUTPUTS)
    if output_act == "sigmoid":
        return rng.uniform(0, 1, (NUM_SAMPLES, NUM_OUTPUTS))
    return rng.normal(0, 1, (NUM_SAMPLES, NUM_OUTPUTS))


def numerical_gradients(model, inputs, targets, eps=1e-6, penalty=None):
    # Central differences of the mean batch loss (plus an optional penalty on the model) in relation to every
    # weight and bias
    def objective():
        loss = loss_fn(model.configs["loss"], targets, model.forward(inputs))
        return loss + (penalty(model) if penalty else 0.0)

    gradients = []

    for layer in model.layers:
        layer_gradients = {}

        for key in ("weights", "biases"):
            params = layer[key]
            grad = np.zeros_like(params)

            for idx in np.ndindex(params.shape):
                original = params[idx]

                params[idx] = original + eps
                loss_plus = objective()

                params[idx] = original - eps
                loss_minus = objective()

                params[idx] = original
                grad[idx] = (loss_plus - loss_minus) / (2 * eps)

            layer_gradients[key] = grad

        gradients.append(layer_gradients)

    return gradients


def applied_gradients(model, inputs, targets, learning_rate=0.5, **train_options):
    # One training step over the whole batch moves each parameter by -learning_rate * gradient.
    # A large learning rate makes any use of already-updated weights show up clearly.
    before = model.get_layers()

    with redirect_stdout(io.StringIO()):
        model.train(inputs, targets, epochs=1, learning_rate=learning_rate, batch_size=len(inputs), **train_options)

    return [
        {key: (old[key] - new[key]) / learning_rate for key in ("weights", "biases")}
        for old, new in zip(before, model.layers)
    ]


class GradientCheckTest(unittest.TestCase):
    CASES = [
        ("relu", "linear", "mse"),
        ("relu", "sigmoid", "mse"),
        ("relu", "softmax", "mse"),  # backpropagates through the softmax Jacobian
        ("sigmoid", "softmax", "categorical_crossentropy"),  # softmax + cross-entropy shortcut
        ("relu", "softmax", "sparse_categorical_crossentropy"),
    ]

    def test_backpropagation_matches_numerical_gradients(self):
        for hidden_act, output_act, loss in self.CASES:
            with self.subTest(hidden=hidden_act, output=output_act, loss=loss):
                np.random.seed(0)  # weight initialization
                rng = np.random.default_rng(1)

                model = build_model(hidden_act, output_act, loss)
                inputs = rng.normal(0, 1, (NUM_SAMPLES, NUM_INPUTS))
                targets = make_targets(loss, output_act, rng)

                self.assert_gradients_match(model, inputs, targets)

    def test_backpropagation_with_input_normalization(self):
        # Unscaled inputs: the backward pass must use the same normalized inputs as the forward pass
        np.random.seed(0)
        rng = np.random.default_rng(4)

        model = build_model("relu", "linear", "mse")
        inputs = rng.normal(0, 1, (NUM_SAMPLES, NUM_INPUTS)) * [50.0, 1.0, 0.1] + [100.0, 0.0, -3.0]
        targets = rng.normal(0, 1, (NUM_SAMPLES, NUM_OUTPUTS))
        model.fit_normalization(inputs)

        self.assert_gradients_match(model, inputs, targets)

    def test_backpropagation_with_l1_and_l2(self):
        # The update adds l1 * sign(W) + l2 * W to the weight gradients, which is the gradient of the penalty
        # l1 * sum(|W|) + l2 / 2 * sum(W^2) over the weights (biases aren't regularized)
        l1_lambda, l2_lambda = 0.03, 0.2

        def penalty(model):
            return sum(l1_lambda * np.abs(l["weights"]).sum() + l2_lambda / 2 * (l["weights"] ** 2).sum() for l in model.layers)

        np.random.seed(0)
        rng = np.random.default_rng(6)

        model = build_model("relu", "linear", "mse")
        inputs = rng.normal(0, 1, (NUM_SAMPLES, NUM_INPUTS))
        targets = rng.normal(0, 1, (NUM_SAMPLES, NUM_OUTPUTS))

        self.assert_gradients_match(model, inputs, targets, penalty=penalty, l1_lambda=l1_lambda, l2_lambda=l2_lambda)

    def assert_gradients_match(self, model, inputs, targets, penalty=None, **train_options):
        expected = numerical_gradients(model, inputs, targets, penalty=penalty)
        actual = applied_gradients(model, inputs, targets, **train_options)

        for layer_index, (exp, act) in enumerate(zip(expected, actual)):
            for key in ("weights", "biases"):
                np.testing.assert_allclose(
                    act[key], exp[key], rtol=1e-4, atol=1e-8,
                    err_msg=f"layer {layer_index + 1} {key}",
                )


class FeedForwardTest(unittest.TestCase):
    def test_softmax_output_is_a_distribution_per_sample(self):
        np.random.seed(0)
        model = build_model("relu", "softmax", "categorical_crossentropy")

        single = model.predict(np.array([1.0, 2.0, 3.0]))
        batch = model.predict(np.random.normal(0, 1, (6, NUM_INPUTS)))

        self.assertAlmostEqual(single.sum(), 1.0)
        self.assertFalse(np.allclose(single, 1.0))
        np.testing.assert_allclose(batch.sum(axis=1), np.ones(6))

    def test_models_do_not_share_layers_or_configs(self):
        first = FeedForward()
        first.add_layer(num_inputs=2, num_neurons=2, act_fn="linear")
        first.configure(loss="mse")

        second = FeedForward()

        self.assertEqual(len(second.layers), 0)
        self.assertEqual(second.configs, {})

    def test_epoch_loss_is_the_mean_over_all_samples(self):
        # 10 samples with batch_size=4 gives batches of 4, 4 and 2. With learning_rate=0 the
        # weights don't change, so the epoch loss must equal the loss over the whole dataset.
        np.random.seed(0)
        rng = np.random.default_rng(2)
        model = build_model("relu", "linear", "mse")
        inputs = rng.normal(0, 1, (10, NUM_INPUTS))
        targets = rng.normal(0, 1, (10, NUM_OUTPUTS))

        with redirect_stdout(io.StringIO()):
            epoch_loss, _ = model.train(inputs, targets, epochs=1, learning_rate=0.0, batch_size=4)

        self.assertAlmostEqual(epoch_loss, loss_fn("mse", targets, model.forward(inputs)))

    def test_train_with_fewer_samples_than_batch_size(self):
        np.random.seed(0)
        model = build_model("relu", "linear", "mse")

        with redirect_stdout(io.StringIO()):
            epoch_loss, _ = model.train(np.ones((3, NUM_INPUTS)), np.zeros((3, NUM_OUTPUTS)), epochs=1, learning_rate=0.01, batch_size=32)

        self.assertTrue(np.isfinite(epoch_loss))

    def test_sparse_and_one_hot_labels_evaluate_the_same(self):
        np.random.seed(0)
        rng = np.random.default_rng(3)
        model = build_model("relu", "softmax", "categorical_crossentropy")
        inputs = rng.normal(0, 1, (20, NUM_INPUTS))
        labels = rng.integers(0, NUM_OUTPUTS, 20)

        one_hot_loss, one_hot_accuracy = model.evaluate(inputs, to_categorical(labels, num_classes=NUM_OUTPUTS))

        model.configure(loss="sparse_categorical_crossentropy")
        sparse_loss, sparse_accuracy = model.evaluate(inputs, labels)

        self.assertAlmostEqual(sparse_loss, one_hot_loss)
        self.assertAlmostEqual(sparse_accuracy, one_hot_accuracy)


class TrainingStepTest(unittest.TestCase):
    def test_each_batch_is_one_update_in_order(self):
        # 10 samples with batch_size=4 are three updates (4, 4 and 2 samples), the same as training on each chunk
        rng = np.random.default_rng(7)
        inputs = rng.normal(0, 1, (10, NUM_INPUTS))
        targets = rng.normal(0, 1, (10, NUM_OUTPUTS))

        np.random.seed(0)
        whole = build_model("relu", "linear", "mse")
        np.random.seed(0)
        chunked = build_model("relu", "linear", "mse")

        with redirect_stdout(io.StringIO()):
            whole.train(inputs, targets, epochs=1, learning_rate=0.1, batch_size=4)

            for start in (0, 4, 8):
                chunk = slice(start, start + 4)
                chunked.train(inputs[chunk], targets[chunk], epochs=1, learning_rate=0.1, batch_size=4)

        for whole_layer, chunked_layer in zip(whole.layers, chunked.layers):
            np.testing.assert_allclose(whole_layer["weights"], chunked_layer["weights"], rtol=1e-12)
            np.testing.assert_allclose(whole_layer["biases"], chunked_layer["biases"], rtol=1e-12)

    def test_loss_gradient_is_clipped(self):
        # One linear neuron starting at w=0, b=0 and one sample x=3, target=1000:
        # dL/dprediction = 2 * (0 - 1000) = -2000 is clipped to -10, so w += lr * 10 * 3 and b += lr * 10
        model = FeedForward()
        model.add_layer(num_inputs=1, num_neurons=1, act_fn="linear")
        model.configure(loss="mse")
        model.layers[0]["weights"][:] = 0.0

        with redirect_stdout(io.StringIO()):
            model.train([[3.0]], [[1000.0]], epochs=1, learning_rate=0.01, batch_size=1)

        np.testing.assert_allclose(model.layers[0]["weights"], [[0.3]])
        np.testing.assert_allclose(model.layers[0]["biases"], [0.1])


class KnownValuesTest(unittest.TestCase):
    # Values computed by hand, so a loss and its derivative can't be wrong in the same way unnoticed
    PREDICTIONS = [[0.2, 0.7, 0.1], [0.5, 0.25, 0.25]]
    ONE_HOT = [[0, 1, 0], [1, 0, 0]]
    CLASSES = [1, 0]

    def test_activations(self):
        np.testing.assert_allclose(softmax([0.0, np.log(2), np.log(3)]), [1/6, 2/6, 3/6])
        np.testing.assert_allclose(softmax([[0.0, 0.0], [np.log(3), 0.0]]), [[0.5, 0.5], [0.75, 0.25]])
        np.testing.assert_allclose(sigmoid([0.0, np.log(3)]), [0.5, 0.75])
        np.testing.assert_allclose(sigmoid_derivative([0.0, np.log(3)]), [0.25, 0.1875])
        np.testing.assert_allclose(relu([-2.0, 0.0, 3.0]), [0.0, 0.0, 3.0])
        np.testing.assert_allclose(relu_derivative([-2.0, 0.0, 3.0]), [0, 0, 1])

    def test_mse(self):
        # ((1 - 1.5)^2 + (2 - 2.5)^2 + (0 - 1)^2 + (4 - 4)^2) / 4 = 0.375
        y_true, y_pred = [[1, 2], [0, 4]], [[1.5, 2.5], [1, 4]]

        self.assertAlmostEqual(loss_fn("mse", y_true, y_pred), 0.375)
        np.testing.assert_allclose(loss_derivative_fn("mse", y_true, y_pred), [[0.25, 0.25], [0.5, 0.0]])  # 2 * (pred - true) / 4

    def test_crossentropy(self):
        expected_loss = (-np.log(0.7) - np.log(0.5)) / 2
        expected_derivative = [[0, -1 / (0.7 * 2), 0], [-1 / (0.5 * 2), 0, 0]]  # -y / p / batch_size

        self.assertAlmostEqual(loss_fn("categorical_crossentropy", self.ONE_HOT, self.PREDICTIONS), expected_loss)
        self.assertAlmostEqual(loss_fn("sparse_categorical_crossentropy", self.CLASSES, self.PREDICTIONS), expected_loss)
        self.assertAlmostEqual(loss_fn("categorical_crossentropy", [0, 1, 0], [0.2, 0.7, 0.1]), -np.log(0.7))  # single sample

        np.testing.assert_allclose(loss_derivative_fn("categorical_crossentropy", self.ONE_HOT, self.PREDICTIONS), expected_derivative)
        np.testing.assert_allclose(loss_derivative_fn("sparse_categorical_crossentropy", self.CLASSES, self.PREDICTIONS), expected_derivative)

    def test_softmax_crossentropy_shortcut(self):
        # (p - y) / batch_size
        expected = [[0.1, -0.15, 0.05], [-0.25, 0.125, 0.125]]

        np.testing.assert_allclose(softmax_crossentropy_delta("categorical_crossentropy", self.ONE_HOT, self.PREDICTIONS), expected)
        np.testing.assert_allclose(softmax_crossentropy_delta("sparse_categorical_crossentropy", self.CLASSES, self.PREDICTIONS), expected)


class MetricsTest(unittest.TestCase):
    def linear_model(self):
        # y = 2x - 1 exactly
        model = FeedForward()
        model.add_layer(num_inputs=1, num_neurons=1, act_fn="linear")
        model.configure(loss="mse")
        model.layers[0]["weights"][:] = 2.0
        model.layers[0]["biases"][:] = -1.0
        return model

    def test_regression_has_no_accuracy(self):
        self.assertIsNone(self.linear_model().evaluate([[0.0], [1.0]], [[-1.0], [1.0]])[1])

    def test_regression_metrics(self):
        # Predictions -1, 1, 3, 5 for targets -1, 1, 4, 5: errors 0, 0, -1, 0
        # mae = 1/4, rmse = sqrt(1/4), r2 = 1 - 1 / sum((targets - 2.25)^2) = 1 - 1 / 22.75
        metrics = self.linear_model().regression_metrics([[0.0], [1.0], [2.0], [3.0]], [-1.0, 1.0, 4.0, 5.0])

        self.assertAlmostEqual(metrics["mae"], 0.25)
        self.assertAlmostEqual(metrics["rmse"], 0.5)
        self.assertAlmostEqual(metrics["r2"], 1 - 1 / 22.75)

    def test_binary_accuracy_with_one_sigmoid_output(self):
        # sigmoid(x) = 0.12, 0.27, 0.73, 0.88 -> classes 0, 0, 1, 1 for labels 0, 1, 1, 1
        model = FeedForward()
        model.add_layer(num_inputs=1, num_neurons=1, act_fn="sigmoid")
        model.configure(loss="mse")
        model.layers[0]["weights"][:] = 1.0

        self.assertAlmostEqual(model.evaluate([[-2.0], [-1.0], [1.0], [2.0]], [0, 1, 1, 1])[1], 0.75)

    def test_validation_metrics_per_epoch(self):
        np.random.seed(0)
        rng = np.random.default_rng(8)
        model = build_model("relu", "softmax", "categorical_crossentropy")
        inputs = rng.normal(0, 1, (30, NUM_INPUTS))
        targets = to_categorical(rng.integers(0, NUM_OUTPUTS, 30), num_classes=NUM_OUTPUTS)
        val_inputs = rng.normal(0, 1, (10, NUM_INPUTS))
        val_targets = to_categorical(rng.integers(0, NUM_OUTPUTS, 10), num_classes=NUM_OUTPUTS)

        with redirect_stdout(io.StringIO()):
            model.train(inputs, targets, epochs=3, learning_rate=0.1, batch_size=8, validation_data=(val_inputs, val_targets))

        self.assertEqual([len(model.history[key]) for key in ("loss", "val_loss", "val_accuracy")], [3, 3, 3])

        # The last values are the metrics of the final model
        val_loss, val_accuracy = model.evaluate(val_inputs, val_targets)
        self.assertAlmostEqual(model.history["val_loss"][-1], val_loss)
        self.assertAlmostEqual(model.history["val_accuracy"][-1], val_accuracy)


class DatasetShapeTest(unittest.TestCase):
    # Single input and single output: y = 2x - 1
    def train_linear(self, inputs, targets):
        np.random.seed(0)
        model = FeedForward()
        model.add_layer(num_inputs=1, num_neurons=1, act_fn="linear")
        model.configure(loss="mse")

        with redirect_stdout(io.StringIO()):
            model.train(inputs, targets, epochs=100, learning_rate=0.01, batch_size=8)

        return model

    def test_flat_lists_for_one_input_and_one_output(self):
        x = np.random.default_rng(5).uniform(-5, 5, 40)
        y = 2 * x - 1

        columns = self.train_linear(x.reshape(-1, 1), y.reshape(-1, 1))
        flat = self.train_linear(list(x), list(y))

        np.testing.assert_allclose(flat.layers[0]["weights"], columns.layers[0]["weights"])
        np.testing.assert_allclose(flat.layers[0]["biases"], columns.layers[0]["biases"])
        np.testing.assert_allclose(flat.layers[0]["weights"], [[2.0]], atol=1e-3)
        np.testing.assert_allclose(flat.layers[0]["biases"], [-1.0], atol=1e-3)

        self.assertAlmostEqual(flat.evaluate(x, y)[0], columns.evaluate(x.reshape(-1, 1), y.reshape(-1, 1))[0])
        self.assertLess(flat.evaluate(x, y)[0], 1e-6)

    def test_mismatched_shapes_raise_instead_of_broadcasting(self):
        np.random.seed(0)
        model = build_model("relu", "linear", "mse")  # 3 inputs, 3 outputs

        # One value per sample for 3 outputs. With 3 samples, (3, 3) - (3,) broadcasts without any NumPy error
        inputs = np.ones((NUM_OUTPUTS, NUM_INPUTS))
        flat_targets = np.ones(NUM_OUTPUTS)

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                model.train(inputs, flat_targets, epochs=1, learning_rate=0.01)

            with self.assertRaises(ValueError):
                model.train(np.ones((4, 2)), np.ones((4, NUM_OUTPUTS)), epochs=1, learning_rate=0.01)  # 2 features for 3 inputs

        with self.assertRaises(ValueError):
            model.evaluate(inputs, flat_targets)


class SaveAndSeedTest(unittest.TestCase):
    def test_save_keeps_only_the_parameters(self):
        np.random.seed(0)
        model = build_model("relu", "linear", "mse")
        inputs = np.random.normal(0, 1, (500, NUM_INPUTS))
        expected = model.predict(inputs)  # leaves the activations of a 500-sample batch in the layers

        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "model.joblib")
            model.save(filepath)
            saved_layers = load(filepath)["layers"]
            loaded = load_model(filepath)

        self.assertEqual([sorted(layer) for layer in saved_layers], [["act_fn", "biases", "weights"]] * 2)
        np.testing.assert_allclose(loaded.predict(inputs), expected)

    def test_same_seed_gives_the_same_training(self):
        def train_with_seed(seed):
            set_seed(seed)
            model = build_model("relu", "linear", "mse")
            inputs = np.random.normal(0, 1, (20, NUM_INPUTS))
            targets = np.random.normal(0, 1, (20, NUM_OUTPUTS))

            with redirect_stdout(io.StringIO()):
                model.train(inputs, targets, epochs=2, learning_rate=0.01, batch_size=8)

            return model.layers[0]["weights"]

        np.testing.assert_array_equal(train_with_seed(7), train_with_seed(7))
        self.assertFalse(np.array_equal(train_with_seed(7), train_with_seed(8)))


class NormalizationTest(unittest.TestCase):
    def test_inputs_are_scaled_with_the_training_range(self):
        model = FeedForward()
        model.fit_normalization([[0.0, 10.0], [4.0, 30.0]])

        np.testing.assert_allclose(model.normalize([[2.0, 10.0], [8.0, 40.0]]), [[0.5, 0.0], [2.0, 1.5]])

    def test_constant_feature_does_not_divide_by_zero(self):
        model = FeedForward()
        model.fit_normalization([[1.0, 5.0], [3.0, 5.0]])

        np.testing.assert_allclose(model.normalize([[2.0, 5.0]]), [[0.5, 0.0]])

    def test_models_without_normalization_receive_the_inputs_unchanged(self):
        np.testing.assert_allclose(FeedForward().normalize([[100.0, -3.0]]), [[100.0, -3.0]])

    def test_normalization_is_saved_with_the_model(self):
        np.random.seed(0)
        model = build_model("relu", "softmax", "categorical_crossentropy")
        model.fit_normalization(np.random.uniform(-5, 100, (20, NUM_INPUTS)))
        raw_inputs = np.array([[20.0, 70.0, 5.0], [-3.0, 90.0, 12.0]])

        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "model.joblib")
            model.save(filepath)
            loaded = load_model(filepath)

        np.testing.assert_allclose(loaded.predict(raw_inputs), model.predict(raw_inputs))
        np.testing.assert_allclose(clone_model(model).predict(raw_inputs), model.predict(raw_inputs))


if __name__ == "__main__":
    unittest.main()
