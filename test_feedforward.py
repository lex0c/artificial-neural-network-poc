import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
from joblib import load

from activations import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax, tanh, tanh_derivative
from data import set_seed, to_categorical
from feedforward import FeedForward, load_model, clone_model
from losses import loss_derivative_fn, loss_fn, sigmoid_binary_crossentropy_delta, softmax_crossentropy_delta
from optimizers import Adam, Momentum, RMSprop, SGD


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
    if loss == "binary_crossentropy":
        return rng.integers(0, 2, (NUM_SAMPLES, NUM_OUTPUTS)).astype(float)  # a yes/no answer per output
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
        ("tanh", "linear", "mse"),
        ("relu", "softmax", "mse"),  # backpropagates through the softmax Jacobian
        ("sigmoid", "softmax", "categorical_crossentropy"),  # softmax + cross-entropy shortcut
        ("relu", "softmax", "sparse_categorical_crossentropy"),
        ("relu", "sigmoid", "binary_crossentropy"),  # sigmoid + binary cross-entropy shortcut
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
            history = model.train(inputs, targets, epochs=1, learning_rate=0.0, batch_size=4)

        self.assertAlmostEqual(history["loss"][0], loss_fn("mse", targets, model.forward(inputs)))

    def test_train_with_fewer_samples_than_batch_size(self):
        np.random.seed(0)
        model = build_model("relu", "linear", "mse")

        with redirect_stdout(io.StringIO()):
            history = model.train(np.ones((3, NUM_INPUTS)), np.zeros((3, NUM_OUTPUTS)), epochs=1, learning_rate=0.01, batch_size=32)

        self.assertTrue(np.isfinite(history["loss"][0]))

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
            whole.train(inputs, targets, epochs=1, learning_rate=0.1, batch_size=4, shuffle=False)

            for start in (0, 4, 8):
                chunk = slice(start, start + 4)
                chunked.train(inputs[chunk], targets[chunk], epochs=1, learning_rate=0.1, batch_size=4, shuffle=False)

        for whole_layer, chunked_layer in zip(whole.layers, chunked.layers):
            np.testing.assert_allclose(whole_layer["weights"], chunked_layer["weights"], rtol=1e-12)
            np.testing.assert_allclose(whole_layer["biases"], chunked_layer["biases"], rtol=1e-12)

    def test_each_epoch_trains_on_a_random_order_of_the_samples(self):
        rng = np.random.default_rng(9)
        inputs = rng.normal(0, 1, (10, NUM_INPUTS))
        targets = rng.normal(0, 1, (10, NUM_OUTPUTS))

        np.random.seed(0)
        shuffled = build_model("relu", "linear", "mse")
        np.random.seed(0)
        permuted = build_model("relu", "linear", "mse")

        with redirect_stdout(io.StringIO()):
            np.random.seed(1)
            shuffled.train(inputs, targets, epochs=1, learning_rate=0.1, batch_size=4)

            np.random.seed(1)
            order = np.random.permutation(10)
            permuted.train(inputs[order], targets[order], epochs=1, learning_rate=0.1, batch_size=4, shuffle=False)

        for shuffled_layer, permuted_layer in zip(shuffled.layers, permuted.layers):
            np.testing.assert_allclose(shuffled_layer["weights"], permuted_layer["weights"], rtol=1e-12)

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
        np.testing.assert_allclose(tanh([0.0, np.log(3)]), [0.0, 0.8])          # tanh(ln 3) = (9 - 1) / (9 + 1)
        np.testing.assert_allclose(tanh_derivative([0.0, np.log(3)]), [1.0, 0.36])  # 1 - tanh^2

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


    def test_binary_crossentropy(self):
        # Answers yes (1) with probability 0.8 and no (0) with probability 0.3 of yes: -(log 0.8 + log 0.7) / 2
        y_true, y_pred = [1, 0], [0.8, 0.3]

        self.assertAlmostEqual(loss_fn("binary_crossentropy", y_true, y_pred), -(np.log(0.8) + np.log(0.7)) / 2)
        # (p - y) / (p (1 - p)) / 2
        np.testing.assert_allclose(loss_derivative_fn("binary_crossentropy", y_true, y_pred), [-0.2 / 0.16 / 2, 0.3 / 0.21 / 2])

    def test_sigmoid_binary_crossentropy_shortcut(self):
        # (p - y) / number of values, the derivative of the loss times the derivative of the sigmoid, p (1 - p)
        y_true, y_pred = np.array([[1.0, 0.0]]), np.array([[0.8, 0.3]])
        chain_rule = loss_derivative_fn("binary_crossentropy", y_true, y_pred) * y_pred * (1 - y_pred)

        np.testing.assert_allclose(sigmoid_binary_crossentropy_delta(y_true, y_pred), [[-0.1, 0.15]])
        np.testing.assert_allclose(sigmoid_binary_crossentropy_delta(y_true, y_pred), chain_rule)

    def test_binary_crossentropy_corrects_a_confident_mistake(self):
        # A sigmoid output answers 0.0025 (z = -6) for a sample whose answer is 1. With mse, the gradient of z is
        # 2 (p - y) * p (1 - p), almost 0 because the sigmoid is flat there; with binary cross-entropy it is p - y.
        deltas = {}

        for loss in ("mse", "binary_crossentropy"):
            model = FeedForward()
            model.add_layer(num_inputs=1, num_neurons=1, act_fn="sigmoid")
            model.configure(loss=loss)
            model.layers[0]["weights"][:] = 0.0
            model.layers[0]["biases"][:] = -6.0
            deltas[loss] = model.output_delta(np.array([[1.0]]), model.forward_layers(np.array([[0.0]])))[0, 0]

        self.assertAlmostEqual(deltas["binary_crossentropy"], sigmoid(-6.0) - 1)
        self.assertLess(abs(deltas["mse"]), abs(deltas["binary_crossentropy"]) / 100)


class WorkedExampleTest(unittest.TestCase):
    def test_worked_example_in_feedforward_md(self):
        # The training step worked out by hand in FEEDFORWARD.md: y = w * x + b with w = 0.5, b = 0,
        # one sample x = 3 with target 5, mean squared error and learning rate 0.01
        model = FeedForward()
        model.add_layer(num_inputs=1, num_neurons=1, act_fn="linear")
        model.configure(loss="mse")
        model.layers[0]["weights"][:] = 0.5
        x, y = np.array([[3.0]]), np.array([[5.0]])

        y_hat = model.forward_layers(x)
        np.testing.assert_allclose(y_hat, [[1.5]])                    # 1. forward
        self.assertAlmostEqual(loss_fn("mse", y, y_hat), 12.25)      # 2. loss

        delta = model.output_delta(y, y_hat)
        np.testing.assert_allclose(delta, [[-7.0]])                   # 3. output delta

        [(grad_weights, grad_biases)] = model.backward(delta)
        np.testing.assert_allclose(grad_weights, [[-21.0]])           # 4. backward
        np.testing.assert_allclose(grad_biases, [-7.0])

        model.update([(grad_weights, grad_biases)], learning_rate=0.01)
        np.testing.assert_allclose(model.layers[0]["weights"], [[0.71]])  # 5. update
        np.testing.assert_allclose(model.layers[0]["biases"], [0.07])

        self.assertAlmostEqual(loss_fn("mse", y, model.forward_layers(x)), 7.84)

        # train_step runs the same five steps
        model.layers[0]["weights"][:] = 0.5
        model.layers[0]["biases"][:] = 0.0
        self.assertAlmostEqual(model.train_step(x, y, learning_rate=0.01), 12.25)
        np.testing.assert_allclose(model.layers[0]["weights"], [[0.71]])


class OptimizerTest(unittest.TestCase):
    def one_parameter(self):
        return [{"weights": np.array([[1.0]]), "biases": np.array([0.0])}]

    def test_adam_steps_by_hand(self):
        # Gradients 2 and then 4, learning rate 0.1, beta1 0.9, beta2 0.999:
        #   t=1: m = 0.2, v = 0.004, corrected m = 2, v = 4: the step is 0.1 * 2 / sqrt(4) = 0.1
        #   t=2: m = 0.58, v = 0.019996, corrected m = 3.0526, v = 10.003: the step is 0.1 * 3.0526 / 3.1628 = 0.0965
        layers = self.one_parameter()
        adam = Adam()

        adam.update(layers, [(np.array([[2.0]]), np.array([2.0]))], learning_rate=0.1)
        np.testing.assert_allclose(layers[0]["weights"], [[0.9000000005]], rtol=1e-12)

        adam.update(layers, [(np.array([[4.0]]), np.array([4.0]))], learning_rate=0.1)
        np.testing.assert_allclose(layers[0]["weights"], [[0.8034817982229947]], rtol=1e-12)

    def test_momentum_steps_by_hand(self):
        # Gradients 2 and then 4, learning rate 0.1, beta 0.9:
        #   velocity 2: the weight goes from 1 to 1 - 0.1 * 2 = 0.8
        #   velocity 0.9 * 2 + 4 = 5.8: it goes to 0.8 - 0.1 * 5.8 = 0.22
        layers = self.one_parameter()
        momentum = Momentum()

        momentum.update(layers, [(np.array([[2.0]]), np.array([2.0]))], learning_rate=0.1)
        np.testing.assert_allclose(layers[0]["weights"], [[0.8]])

        momentum.update(layers, [(np.array([[4.0]]), np.array([4.0]))], learning_rate=0.1)
        np.testing.assert_allclose(layers[0]["weights"], [[0.22]])

    def test_rmsprop_steps_by_hand(self):
        # Gradients 2 and then 4, learning rate 0.1, rho 0.9:
        #   mean square 0.1 * 4 = 0.4: the step is 0.1 * 2 / sqrt(0.4) = 0.31623
        #   mean square 0.9 * 0.4 + 0.1 * 16 = 1.96: the step is 0.1 * 4 / 1.4 = 0.28571
        layers = self.one_parameter()
        rmsprop = RMSprop()

        rmsprop.update(layers, [(np.array([[2.0]]), np.array([2.0]))], learning_rate=0.1)
        np.testing.assert_allclose(layers[0]["weights"], [[1 - 0.2 / np.sqrt(0.4)]])

        rmsprop.update(layers, [(np.array([[4.0]]), np.array([4.0]))], learning_rate=0.1)
        np.testing.assert_allclose(layers[0]["weights"], [[1 - 0.2 / np.sqrt(0.4) - 0.4 / 1.4]])

    def test_first_adam_step_doesnt_depend_on_the_gradient_scale(self):
        # SGD moves each parameter in proportion to its gradient; the first Adam step is about the learning rate
        gradients = [(np.array([[1e-4, 1e3]]), np.array([-5.0]))]

        sgd_layers = [{"weights": np.zeros((1, 2)), "biases": np.zeros(1)}]
        SGD().update(sgd_layers, gradients, learning_rate=0.01)
        np.testing.assert_allclose(sgd_layers[0]["weights"], [[-1e-6, -10.0]])

        adam_layers = [{"weights": np.zeros((1, 2)), "biases": np.zeros(1)}]
        Adam().update(adam_layers, gradients, learning_rate=0.01)
        np.testing.assert_allclose(adam_layers[0]["weights"], [[-0.01, -0.01]], rtol=1e-3)
        np.testing.assert_allclose(adam_layers[0]["biases"], [0.01], rtol=1e-6)

    def test_adam_state_carries_over_between_train_calls(self):
        rng = np.random.default_rng(10)
        inputs = rng.normal(0, 1, (8, NUM_INPUTS))
        targets = rng.normal(0, 1, (8, NUM_OUTPUTS))

        def trained(calls, reconfigure=False):
            np.random.seed(0)
            model = build_model("relu", "linear", "mse")
            model.configure(loss="mse", optimizer="adam")

            with redirect_stdout(io.StringIO()):
                for epochs in calls:
                    if reconfigure:
                        model.configure(loss="mse", optimizer="adam")  # starts a new optimizer
                    model.train(inputs, targets, epochs=epochs, learning_rate=0.01, batch_size=4, shuffle=False)

            return model.layers[0]["weights"]

        np.testing.assert_allclose(trained([1, 1]), trained([2]), rtol=1e-12)
        self.assertFalse(np.allclose(trained([1, 1], reconfigure=True), trained([2]), rtol=1e-12))

    def test_optimizer_is_saved_with_the_model(self):
        np.random.seed(0)
        model = build_model("relu", "linear", "mse")
        model.configure(loss="mse", optimizer="adam")

        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "model.joblib")
            model.save(filepath)
            loaded = load_model(filepath)

        self.assertEqual(loaded.configs["optimizer"], "adam")
        self.assertIsInstance(loaded.optimizer, Adam)
        self.assertEqual(loaded.optimizer.t, 0)  # the averages of the gradients aren't saved

    def test_models_without_an_optimizer_use_sgd(self):
        self.assertIsInstance(FeedForward().optimizer, SGD)

        with self.assertRaises(ValueError):
            FeedForward().configure(loss="mse", optimizer="adagrad")  # not one of optimizers.py


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

    def test_binary_accuracy_counts_each_answer(self):
        # Two yes/no answers per sample (binary_crossentropy): 3 of the 4 answers are right
        model = FeedForward()
        model.add_layer(num_inputs=2, num_neurons=2, act_fn="sigmoid")
        model.configure(loss="binary_crossentropy")
        predictions, labels = np.array([[0.9, 0.2], [0.4, 0.7]]), np.array([[1, 0], [1, 1]])

        self.assertAlmostEqual(model.accuracy(predictions, labels), 0.75)

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


class LossAndOutputTest(unittest.TestCase):
    def test_crossentropy_needs_its_output_activation(self):
        for output_act, loss in (("sigmoid", "categorical_crossentropy"), ("linear", "sparse_categorical_crossentropy"),
                                 ("softmax", "binary_crossentropy"), ("relu", "binary_crossentropy")):
            with self.subTest(output=output_act, loss=loss), self.assertRaises(ValueError):
                build_model("relu", output_act, loss)

    def test_checked_again_when_training(self):
        # configure before the output layer exists can't check it, so train does
        model = FeedForward()
        model.configure(loss="categorical_crossentropy")
        model.add_layer(num_inputs=1, num_neurons=1, act_fn="sigmoid")

        with self.assertRaises(ValueError):
            model.train([[0.0]], [[1.0]], epochs=1, learning_rate=0.1)


class EarlyStoppingTest(unittest.TestCase):
    def train(self, val_losses, epochs, patience):
        # The validation losses of the epochs are given: evaluate is replaced, and keeps the weights after each epoch
        np.random.seed(0)
        rng = np.random.default_rng(9)
        model = build_model("relu", "linear", "mse")
        weights_after = []

        def evaluate(inputs, targets):
            weights_after.append(model.get_layers())
            return val_losses[len(weights_after) - 1], None

        with mock.patch.object(model, "evaluate", side_effect=evaluate), redirect_stdout(io.StringIO()):
            model.train(rng.normal(0, 1, (8, NUM_INPUTS)), rng.normal(0, 1, (8, NUM_OUTPUTS)), epochs=epochs,
                        learning_rate=0.1, batch_size=4, validation_data=([[0.0] * NUM_INPUTS], [[0.0] * NUM_OUTPUTS]),
                        patience=patience)

        return model, weights_after

    def test_stops_after_patience_epochs_without_improvement(self):
        # Best at epoch 2 (0.5); epochs 3 and 4 aren't better, so with patience 2 it stops after epoch 4
        model, weights_after = self.train([1.0, 0.5, 0.7, 0.6, 0.4, 0.3], epochs=6, patience=2)

        self.assertEqual(len(model.history["loss"]), 4)
        self.assertEqual(model.history["best_epoch"], 2)
        np.testing.assert_array_equal(model.layers[0]["weights"], weights_after[1][0]["weights"])  # back to epoch 2

    def test_restores_the_best_epoch_without_stopping(self):
        model, weights_after = self.train([1.0, 0.5, 0.7], epochs=3, patience=5)

        self.assertEqual(len(model.history["loss"]), 3)
        self.assertEqual(model.history["best_epoch"], 2)
        np.testing.assert_array_equal(model.layers[0]["weights"], weights_after[1][0]["weights"])

    def test_needs_validation_data(self):
        model = build_model("relu", "linear", "mse")

        with self.assertRaises(ValueError):
            model.train([[0.0] * NUM_INPUTS], [[0.0] * NUM_OUTPUTS], epochs=1, learning_rate=0.1, patience=2)


class DropoutTest(unittest.TestCase):
    def model(self, dropout):
        np.random.seed(0)
        model = FeedForward()
        model.add_layer(num_inputs=NUM_INPUTS, num_neurons=NUM_HIDDEN, act_fn="relu", dropout=dropout)
        model.add_layer(num_inputs=NUM_HIDDEN, num_neurons=NUM_OUTPUTS, act_fn="softmax", dropout=dropout)
        model.configure(loss="categorical_crossentropy")
        return model

    def test_backpropagation_matches_numerical_gradients_with_the_same_mask(self):
        # The same seed before every forward pass draws the same inputs to drop, so the loss is a fixed function of the
        # weights, and its numerical gradient is what backward must give for that mask
        model = self.model(dropout=0.5)
        rng = np.random.default_rng(10)
        inputs = rng.normal(0, 1, (NUM_SAMPLES, NUM_INPUTS))
        targets = to_categorical(rng.integers(0, NUM_OUTPUTS, NUM_SAMPLES), num_classes=NUM_OUTPUTS)

        def loss():
            np.random.seed(5)
            return loss_fn("categorical_crossentropy", targets, model.forward_layers(inputs, training=True))

        np.random.seed(5)
        predictions = model.forward_layers(inputs, training=True)
        self.assertTrue(any((cache["mask"] == 0).any() for cache in model.cache))  # some inputs were dropped
        actual = model.backward(model.output_delta(targets, predictions))

        for layer, (grad_weights, grad_biases) in zip(model.layers, actual):
            for params, grad in ((layer["weights"], grad_weights), (layer["biases"], grad_biases)):
                for idx in np.ndindex(params.shape):
                    original = params[idx]
                    params[idx] = original + 1e-6
                    loss_plus = loss()
                    params[idx] = original - 1e-6
                    loss_minus = loss()
                    params[idx] = original
                    self.assertAlmostEqual(grad[idx], (loss_plus - loss_minus) / 2e-6, delta=1e-6)

    def test_drops_and_scales_the_inputs_only_while_training(self):
        model = self.model(dropout=0.5)
        inputs = np.ones((1000, NUM_INPUTS))

        model.forward_layers(inputs, training=True)
        kept = model.cache[0]["inputs"]
        self.assertEqual(set(np.unique(kept)), {0.0, 2.0})  # dropped, or kept and scaled by 1 / (1 - 0.5)
        self.assertAlmostEqual(kept.mean(), 1.0, delta=0.05)  # the expected value of each input is unchanged

        # Predictions use every input as it is: the same as the model without dropout
        without = FeedForward(layers=[{key: l[key] for key in ("weights", "biases", "act_fn")} for l in model.layers],
                              configs=model.get_configs())
        np.testing.assert_array_equal(model.predict(inputs[:5]), without.predict(inputs[:5]))

    def test_training_applies_dropout(self):
        model = self.model(dropout=0.5)

        with mock.patch.object(model, "forward_layers", wraps=model.forward_layers) as forward, redirect_stdout(io.StringIO()):
            model.train(np.ones((4, NUM_INPUTS)), to_categorical([0, 1, 2, 0], num_classes=NUM_OUTPUTS), epochs=1, learning_rate=0.1)

        self.assertEqual(forward.call_args.kwargs, {"training": True})

    def test_dropout_must_be_below_1(self):
        for dropout in (-0.1, 1.0):
            with self.subTest(dropout=dropout), self.assertRaises(ValueError):
                FeedForward().add_layer(num_inputs=2, num_neurons=2, act_fn="relu", dropout=dropout)

    def test_dropout_is_saved_with_the_model(self):
        model = self.model(dropout=0.2)

        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "model.joblib")
            model.save(filepath)
            self.assertEqual([layer["dropout"] for layer in load_model(filepath).layers], [0.2, 0.2])


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
