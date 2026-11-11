import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from feedforward import FeedForward
from etc import loss_fn, to_categorical


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


def numerical_gradients(model, inputs, targets, eps=1e-6):
    # Central differences of the mean batch loss in relation to every weight and bias
    gradients = []

    for layer in model.layers:
        layer_gradients = {}

        for key in ("weights", "biases"):
            params = layer[key]
            grad = np.zeros_like(params)

            for idx in np.ndindex(params.shape):
                original = params[idx]

                params[idx] = original + eps
                loss_plus = loss_fn(model.configs["loss"], targets, model.forward(inputs))

                params[idx] = original - eps
                loss_minus = loss_fn(model.configs["loss"], targets, model.forward(inputs))

                params[idx] = original
                grad[idx] = (loss_plus - loss_minus) / (2 * eps)

            layer_gradients[key] = grad

        gradients.append(layer_gradients)

    return gradients


def applied_gradients(model, inputs, targets, use_granular_update, learning_rate=0.5):
    # One training step over the whole batch moves each parameter by -learning_rate * gradient.
    # A large learning rate makes any use of already-updated weights show up clearly.
    before = model.get_layers()

    with redirect_stdout(io.StringIO()):
        model.train(inputs, targets, epochs=1, learning_rate=learning_rate, batch_size=len(inputs), use_granular_update=use_granular_update)

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
            for use_granular_update in (False, True):
                with self.subTest(hidden=hidden_act, output=output_act, loss=loss, granular=use_granular_update):
                    np.random.seed(0)  # weight initialization
                    rng = np.random.default_rng(1)

                    model = build_model(hidden_act, output_act, loss)
                    inputs = rng.normal(0, 1, (NUM_SAMPLES, NUM_INPUTS))
                    targets = make_targets(loss, output_act, rng)

                    expected = numerical_gradients(model, inputs, targets)
                    actual = applied_gradients(model, inputs, targets, use_granular_update)

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


if __name__ == "__main__":
    unittest.main()
