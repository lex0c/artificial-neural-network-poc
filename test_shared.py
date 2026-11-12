import json
import subprocess
import sys
import unittest

import numpy as np

from activations import activation_backward, relu, relu_derivative, sigmoid, softmax, tanh
from gradcheck import numerical_gradients
from losses import LOSSES, sigmoid_binary_crossentropy_delta, softmax_crossentropy_delta
from optimizers import OPTIMIZERS


# The modules that ssnn/ shares with the framework: activations, losses, optimizers, data, progress, gradcheck and cli.
# They must work in float32 and must not bring in feedforward.py, which limits the BLAS threads (see
# limit_blas_threads), or its dependencies.


def run_python(code):
    # In a new process, so the modules loaded and the BLAS threads of this one don't matter
    return json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout)


class SharedImportsTest(unittest.TestCase):
    def test_shared_modules_dont_load_feedforward(self):
        loaded = run_python(
            "import json, sys\n"
            "import activations, cli, data, gradcheck, losses, optimizers, progress\n"
            "print(json.dumps(sorted({'feedforward', 'threadpoolctl', 'joblib'} & set(sys.modules))))")
        self.assertEqual(loaded, [])

    def test_blas_threads_are_limited_by_the_first_model_not_by_the_import(self):
        threads = run_python(
            "import json\n"
            "import numpy\n"  # loads BLAS, with its own number of threads
            "from threadpoolctl import threadpool_info\n"
            "def threads(): return [pool['num_threads'] for pool in threadpool_info()]\n"
            "before = threads()\n"
            "import feedforward\n"
            "after_import = threads()\n"
            "feedforward.FeedForward()\n"
            "print(json.dumps([before, after_import, threads()]))")
        before, after_import, after_model = threads
        self.assertEqual(after_import, before)
        self.assertTrue(after_model and all(n == 1 for n in after_model))


class Float32Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.z = rng.normal(0, 1, (4, 3)).astype(np.float32)
        self.probabilities = softmax(self.z)
        self.classes = np.array([0, 2, 1, 1])  # integer labels
        self.one_hot = np.eye(3, dtype=int)[self.classes]

    def test_activations_keep_float32(self):
        for name, values in (("relu", relu(self.z)), ("relu_derivative", relu_derivative(self.z)),
                             ("sigmoid", sigmoid(self.z)), ("tanh", tanh(self.z)), ("softmax", self.probabilities),
                             ("softmax backward", activation_backward("softmax", self.z, self.probabilities, self.z)),
                             ("relu backward", activation_backward("relu", self.z, relu(self.z), self.z))):
            with self.subTest(name):
                self.assertEqual(values.dtype, np.float32)

    def test_losses_keep_float32_with_integer_targets(self):
        targets = {"mse": self.one_hot, "categorical_crossentropy": self.one_hot,
                   "sparse_categorical_crossentropy": self.classes, "binary_crossentropy": self.one_hot}

        for name, (loss, derivative) in LOSSES.items():
            with self.subTest(name):
                self.assertEqual(np.asarray(loss(targets[name], self.probabilities)).dtype, np.float32)
                self.assertEqual(derivative(targets[name], self.probabilities).dtype, np.float32)

        self.assertEqual(softmax_crossentropy_delta("sparse_categorical_crossentropy", self.classes, self.probabilities).dtype, np.float32)
        self.assertEqual(sigmoid_binary_crossentropy_delta(self.one_hot, sigmoid(self.z)).dtype, np.float32)

    def test_optimizers_keep_float32(self):
        for name, optimizer_class in OPTIMIZERS.items():
            with self.subTest(name):
                params = [np.ones((2, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)]
                optimizer = optimizer_class()
                for _ in range(2):
                    optimizer.update(params, [np.full((2, 3), 0.5, dtype=np.float32), np.ones(3, dtype=np.float32)], 0.01)

                self.assertEqual([param.dtype for param in params], [np.float32, np.float32])
                self.assertFalse(np.array_equal(params[0], np.ones((2, 3))))  # they moved
                state = [array for value in vars(optimizer).values() if isinstance(value, list) for array in value]
                self.assertTrue(all(array.dtype == np.float32 for array in state))


class GradcheckTest(unittest.TestCase):
    def test_gradient_of_a_known_function(self):
        # L = sum(w^2) + 3 * b, so dL/dw = 2w and dL/db = 3
        w, b = np.array([[1.0, -2.0], [0.5, 3.0]]), np.array([4.0])
        grads = numerical_gradients(lambda: np.sum(w ** 2) + 3 * b[0], [w, b])

        np.testing.assert_allclose(grads[0], 2 * w, rtol=1e-6)
        np.testing.assert_allclose(grads[1], [3.0], rtol=1e-6)
        np.testing.assert_array_equal(w, [[1.0, -2.0], [0.5, 3.0]])  # put back as they were


if __name__ == "__main__":
    unittest.main()
