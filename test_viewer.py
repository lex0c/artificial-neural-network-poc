import multiprocessing
import queue
import time
import unittest
from unittest import mock

import numpy as np

import viewer
from feedforward import FeedForward


def build_model():
    model = FeedForward()
    model.add_layer(num_inputs=2, num_neurons=3, act_fn="tanh")
    model.add_layer(num_inputs=3, num_neurons=2, act_fn="softmax")
    model.configure(loss=None, optimizer="adam")
    return model


def received(weights):
    # The queue hands its items over in the background: waits a moment for them
    time.sleep(0.2)
    return viewer.newest(weights, None)


class ViewerTest(unittest.TestCase):
    def setUp(self):
        # No window in the tests: the viewer process isn't started
        patcher = mock.patch.object(multiprocessing.context.ForkProcess, "start")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sends_the_weights_before_training_and_after_each_report(self):
        model = build_model()
        live = viewer.LiveViewer("CartPole-v1", model)

        layers, label = received(live.weights)
        self.assertEqual(label, "before training")
        np.testing.assert_array_equal(layers[0]["weights"], model.layers[0]["weights"])

        model.layers[0]["weights"] += 1  # e.g. an update
        model.notify("on_metrics", {"iteration": 3, "mean_episode_reward": 10.0})

        layers, label = received(live.weights)
        self.assertEqual(label, "after iteration 3")
        np.testing.assert_array_equal(layers[0]["weights"], model.layers[0]["weights"])

    def test_sends_a_copy(self):
        model = build_model()
        live = viewer.LiveViewer("CartPole-v1", model)
        model.layers[0]["weights"] += 1  # changed after sending: the viewer keeps what was sent

        layers, _ = received(live.weights)
        self.assertFalse(np.array_equal(layers[0]["weights"], model.layers[0]["weights"]))

    def test_newest_skips_to_the_last_item(self):
        weights = queue.Queue()
        self.assertEqual(viewer.newest(weights, "current"), "current")  # nothing new

        for item in ("a", "b", "c"):
            weights.put(item)
        self.assertEqual(viewer.newest(weights, "current"), "c")


if __name__ == "__main__":
    unittest.main()
