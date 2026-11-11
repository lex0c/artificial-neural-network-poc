import base64
import json
import threading
import time

import numpy as np


# An in-memory picture of models while they train or run, for the dashboard (dashboard.py). It works with any
# FeedForward model, whatever it does (regression, classification, a DQN or PPO agent), because it only observes what
# every model has: its layers, the values of its last forward pass, its updates and the metrics its training reports.
#
#   monitor = Monitor()
#   monitor.attach(model, "classifier")  # from now on, the model tells the monitor about its forward passes and updates
#
# Copying the arrays takes time, so the monitor keeps a new copy at most every min_interval seconds, and only the
# first max_samples samples of a batch.


# Arrays travel to the dashboard as base64-encoded float32, which JavaScript reads as a Float32Array.
def encode_array(array):
    array = np.atleast_1d(np.asarray(array, dtype="<f4"))
    return {"shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode("ascii")}


class ModelObserver:
    def __init__(self, monitor, name):
        self.monitor = monitor
        self.name = name

    def on_forward(self, model):
        self.monitor.record_forward(self.name, model)

    def on_update(self, model, gradients, learning_rate):
        self.monitor.record_update(self.name, model, gradients, learning_rate)

    def on_metrics(self, model, values):
        self.monitor.log(values)


class Monitor:
    def __init__(self, max_samples=32, min_interval=0.1, max_points=2000):
        self.max_samples = max_samples
        self.min_interval = min_interval
        self.max_points = max_points  # older points of each metric are thinned out beyond this
        self.lock = threading.Lock()
        self.models = {}   # name -> what the dashboard shows about the model
        self.metrics = {}  # name -> list of [x, value]
        self.axes = {}     # name -> what its x is (epoch, episode, iteration or step), when the values said it
        self.version = 0   # increases with every change, so the dashboard server knows when to send

    def attach(self, model, name):
        model.observers.append(ModelObserver(self, name))

        with self.lock:
            self.models[name] = {"forwards": 0, "updates": 0, "last_copy": 0.0, "activations": None,
                                 "gradient_norms": None, "learning_rate": None}
            self.copy_parameters(name, model)
            self.version += 1

    # Called by the observer while the model trains or runs (in the thread of the training)

    def record_forward(self, name, model):
        state = self.models[name]
        state["forwards"] += 1
        now = time.time()

        if now - state["last_copy"] < self.min_interval:
            return

        # The inputs, z and output of each layer, for the first samples of the batch (a single sample is one row)
        activations = [{key: np.atleast_2d(cache[key])[:self.max_samples].copy() for key in ("inputs", "z", "output")}
                       for cache in model.cache]

        with self.lock:
            state["activations"] = activations
            state["last_copy"] = now
            self.copy_parameters(name, model)
            self.version += 1

    def record_update(self, name, model, gradients, learning_rate):
        state = self.models[name]
        state["updates"] += 1
        # The size of the gradients of each layer, to see whether they fade or explode on the way back
        norms = [[float(np.linalg.norm(grad_weights)), float(np.linalg.norm(grad_biases))] for grad_weights, grad_biases in gradients]

        with self.lock:
            state["gradient_norms"] = norms
            state["learning_rate"] = learning_rate
            self.version += 1

    def copy_parameters(self, name, model):
        state = self.models[name]
        state["layers"] = [{"act_fn": layer["act_fn"], "weights": layer["weights"].copy(), "biases": layer["biases"].copy()}
                           for layer in model.layers]
        state["configs"] = {"loss": model.configs.get("loss"), "optimizer": model.configs.get("optimizer", "sgd"),
                            "normalization": model.configs.get("normalization", {}).get("method")}

    # Any numeric values, e.g. {"episode": 3, "reward": 21.0}: the first key named epoch, episode, iteration or step
    # is the x axis of the others; without one, they are numbered in order.
    def log(self, values):
        x_key = next((key for key in ("epoch", "episode", "iteration", "step") if key in values), None)

        with self.lock:
            for key, value in values.items():
                if key == x_key or value is None or not np.isfinite(value):
                    continue

                series = self.metrics.setdefault(key, [])
                if x_key:
                    self.axes[key] = x_key
                x = values[x_key] if x_key else (series[-1][0] + 1 if series else 1)
                series.append([float(x), float(value)])

                if len(series) > self.max_points:
                    del series[1::2]  # keeps every other point, so long trainings still fit

            self.version += 1

    # Called by the dashboard server (in its own thread)

    def to_json(self):
        with self.lock:
            models = []

            for name, state in self.models.items():
                layers = [{"act_fn": layer["act_fn"], "weights": encode_array(layer["weights"]), "biases": encode_array(layer["biases"])}
                          for layer in state["layers"]]
                activations = None if state["activations"] is None else [
                    {key: encode_array(values) for key, values in layer.items()} for layer in state["activations"]]

                models.append({"name": name, "layers": layers, "activations": activations, "configs": state["configs"],
                               "forwards": state["forwards"], "updates": state["updates"],
                               "gradient_norms": state["gradient_norms"], "learning_rate": state["learning_rate"]})

            return json.dumps({"version": self.version, "models": models, "metrics": self.metrics, "axes": self.axes})
