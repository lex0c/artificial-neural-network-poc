import multiprocessing
import queue
import signal

import numpy as np


# Shows an agent playing in a window while it trains, without slowing the training down.
#
# Rendering the training environment itself would hold every step to the speed of the animation (50 steps per second
# in LunarLander, so a training of a million steps would take more than 5 hours). Instead, a separate process plays
# episodes in its own environment, always with the newest weights of the model: the viewer observes the model, like
# the dashboard (see notify in feedforward.py), and sends the weights to that process each time the training reports
# its metrics, after each PPO iteration or DQN episode.
#
#   LiveViewer("LunarLander-v3", policy)
#
# The window plays the greedy policy: the action with the highest output of the model, as in rl.evaluate_greedy.
# Closing the window stops the viewer, not the training.


class LiveViewer:
    def __init__(self, env_id, model):
        # fork: the viewer process starts as a copy of this one, so it doesn't run the training script again. Started
        # before any other thread of this process exists (so before the dashboard), as fork copies only this thread.
        context = multiprocessing.get_context("fork")
        self.weights = context.Queue()
        self.weights.cancel_join_thread()  # at exit, doesn't wait for weights the viewer will never read
        context.Process(target=play, args=(env_id, model.configs, self.weights), daemon=True).start()

        self.send(model, "before training")
        model.observers.append(self)

    def on_forward(self, model):
        pass

    def on_update(self, model, gradients, learning_rate):
        pass

    def on_metrics(self, model, values):
        step = next((f"{key} {values[key]}" for key in ("iteration", "episode", "epoch") if key in values), "the last update")
        self.send(model, f"after {step}")

    # Never waits: the queue sends in the background, and the viewer skips to the newest weights when it reads them
    def send(self, model, label):
        self.weights.put((model.get_layers(), label))


# The last item in the queue, or current when it is empty
def newest(weights, current):
    try:
        while True:
            current = weights.get_nowait()
    except queue.Empty:
        return current


# Runs in the viewer process: one episode after the other, each one with the newest weights received
def play(env_id, configs, weights):
    import gymnasium as gym
    import pygame

    from feedforward import FeedForward

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is for the training: the viewer ends with it
    env = gym.make(env_id, render_mode="human")
    received = weights.get()  # waits for the first weights

    while True:
        layers, label = received
        model = FeedForward(layers=layers, configs=configs)

        state, _ = env.reset()
        pygame.display.set_caption(f"{env_id}: greedy policy {label}")
        done = False

        while not done:
            action = int(np.argmax(model.predict(state[None])))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Takes the weights sent meanwhile at every step, so they don't pile up while an episode plays (the queue
            # hands them over one at a time): the next episode starts with the newest
            received = newest(weights, received)

            if pygame.event.get(pygame.QUIT):  # the window was closed
                env.close()
                return
