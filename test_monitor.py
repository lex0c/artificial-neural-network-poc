import base64
import io
import json
import unittest
import urllib.request
from contextlib import redirect_stdout

import numpy as np

from dashboard import Dashboard
from feedforward import FeedForward
from monitor import Monitor


def build_model():
    np.random.seed(0)
    model = FeedForward()
    model.add_layer(num_inputs=3, num_neurons=4, act_fn="relu")
    model.add_layer(num_inputs=4, num_neurons=2, act_fn="softmax")
    model.configure(loss="categorical_crossentropy", optimizer="adam")
    return model


def decode(array):
    return np.frombuffer(base64.b64decode(array["data"]), dtype="<f4").reshape(array["shape"])


class MonitorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.inputs = rng.normal(0, 1, (10, 3))
        self.targets = np.eye(2)[rng.integers(0, 2, 10)]

    def test_forward_keeps_the_first_samples_of_each_layer(self):
        model = build_model()
        monitor = Monitor(max_samples=5, min_interval=0)
        monitor.attach(model, "net")

        model.predict(self.inputs)
        state = monitor.models["net"]

        self.assertEqual(state["forwards"], 1)
        for cache, kept in zip(model.cache, state["activations"]):
            for key in ("inputs", "z", "output"):
                np.testing.assert_array_equal(kept[key], cache[key][:5])

    def test_copies_are_throttled(self):
        model = build_model()
        monitor = Monitor(min_interval=1000)
        monitor.attach(model, "net")

        model.predict(self.inputs)
        first = monitor.models["net"]["activations"][0]["inputs"]
        model.predict(self.inputs + 1)

        self.assertEqual(monitor.models["net"]["forwards"], 2)  # counted...
        self.assertIs(monitor.models["net"]["activations"][0]["inputs"], first)  # ...but not copied again

    def test_updates_and_metrics_of_a_training(self):
        model = build_model()
        monitor = Monitor(min_interval=0)
        monitor.attach(model, "net")

        with redirect_stdout(io.StringIO()):
            model.train(self.inputs, self.targets, epochs=3, learning_rate=0.01, batch_size=5)

        state = monitor.models["net"]
        self.assertEqual(state["updates"], 6)  # 3 epochs of 2 batches
        self.assertEqual(state["learning_rate"], 0.01)
        self.assertEqual(len(state["gradient_norms"]), 2)  # one (weights, biases) pair per layer
        self.assertEqual([x for x, _ in monitor.metrics["loss"]], [1.0, 2.0, 3.0])  # one point per epoch
        np.testing.assert_allclose([v for _, v in monitor.metrics["loss"]], model.history["loss"])
        self.assertNotIn("val_loss", monitor.metrics)  # None values are skipped

    def test_observing_doesnt_change_the_training(self):
        def trained(observed):
            model = build_model()
            if observed:
                Monitor(min_interval=0).attach(model, "net")
            np.random.seed(1)
            with redirect_stdout(io.StringIO()):
                model.train(self.inputs, self.targets, epochs=5, learning_rate=0.01, batch_size=4)
            return [layer["weights"] for layer in model.layers]

        for plain, observed in zip(trained(False), trained(True)):
            np.testing.assert_array_equal(plain, observed)

    def test_json_has_the_arrays(self):
        model = build_model()
        monitor = Monitor(min_interval=0)
        monitor.attach(model, "net")
        model.predict(self.inputs)
        monitor.log({"episode": 1, "reward": 5.0})

        state = json.loads(monitor.to_json())
        sent = state["models"][0]

        np.testing.assert_array_equal(decode(sent["layers"][0]["weights"]), model.layers[0]["weights"].astype("<f4"))
        np.testing.assert_array_equal(decode(sent["activations"][1]["output"]), model.cache[1]["output"][:32].astype("<f4"))
        self.assertEqual(state["metrics"], {"reward": [[1.0, 5.0]]})
        self.assertEqual(state["axes"], {"reward": "episode"})


class DashboardTest(unittest.TestCase):
    def test_serves_the_page_the_state_and_the_events(self):
        model = build_model()
        monitor = Monitor(min_interval=0)
        monitor.attach(model, "net")
        model.predict(np.ones((2, 3)))

        dashboard = Dashboard(monitor, port=8765)
        address = dashboard.start()
        try:
            page = urllib.request.urlopen(address + "/", timeout=5).read().decode()
            self.assertIn("Model Dashboard", page)

            state = json.loads(urllib.request.urlopen(address + "/state", timeout=5).read())
            self.assertEqual(state["models"][0]["name"], "net")

            events = urllib.request.urlopen(address + "/events", timeout=5)
            first_line = events.readline().decode()
            self.assertTrue(first_line.startswith("data: {"))
            events.close()
        finally:
            dashboard.stop()


if __name__ == "__main__":
    unittest.main()
