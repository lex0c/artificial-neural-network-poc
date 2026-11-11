import csv
import io
import os
import random
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dqn import evaluate_greedy, learn_from_memory, replay_experience, train_dqn
from feedforward import FeedForward, clone_model


GAMMA = 0.5


class TableModel:
    # Fake model whose Q-values come from a table indexed by the first value of the state
    def __init__(self, table):
        self.table = table

    def predict(self, states):
        return np.array([self.table[int(state[0])] for state in np.atleast_2d(states)], dtype=float)


def state(state_id):
    return np.array([[state_id, 0.0, 0.0, 0.0]])


# The online model prefers action 1 in the next state 10, while the target model prefers action 0
ONLINE = TableModel({0: [0.5, 0.2], 1: [0.7, 0.1], 2: [0.3, 0.4], 10: [1.0, 3.0], 11: [9.0, 9.0]})
TARGET = TableModel({0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0], 10: [5.0, 2.0], 11: [9.0, 9.0]})

# (state, action, reward, next_state, terminal)
TRANSITIONS = {
    0: (state(0), 0, 1.0, state(10), False),  # still running (or truncated by the time limit): bootstraps
    1: (state(1), 1, 1.0, state(11), True),   # the pole fell: no future value
    2: (state(2), 1, 1.0, state(10), False),
}


def targets_by_state(double_dqn):
    # The memory needs more transitions than the batch size, so each sample picks 2 of the 3 transitions.
    # Sampling with several seeds checks all of them; the transitions are identified by their state.
    found = {}

    for seed in range(20):
        random.seed(seed)
        memory = deque(TRANSITIONS.values())
        states, actions, target_q_values = replay_experience(memory, ONLINE, TARGET, batch_size=len(memory) - 1, gamma=GAMMA, double_dqn=double_dqn)
        found.update({int(s[0]): (a, t) for s, a, t in zip(states, actions, target_q_values)})

    return found


class ReplayExperienceTest(unittest.TestCase):
    def assert_targets(self, double_dqn, next_value_of_state_10):
        targets = targets_by_state(double_dqn)
        self.assertEqual(set(targets), set(TRANSITIONS))

        for state_id, (action, target_q) in targets.items():
            _, expected_action, reward, _, terminal = TRANSITIONS[state_id]
            self.assertEqual(action, expected_action)
            self.assertAlmostEqual(target_q, reward if terminal else reward + GAMMA * next_value_of_state_10, msg=f"state {state_id}")

    def test_dqn_uses_the_max_of_the_target_model(self):
        self.assert_targets(double_dqn=False, next_value_of_state_10=5.0)

    def test_double_dqn_evaluates_the_online_choice_with_the_target_model(self):
        # Online argmax in state 10 is action 1, whose target value is 2.0
        self.assert_targets(double_dqn=True, next_value_of_state_10=2.0)

    def test_needs_more_transitions_than_the_batch_size(self):
        memory = deque(TRANSITIONS.values())
        self.assertIsNone(replay_experience(memory, ONLINE, TARGET, batch_size=len(memory), gamma=GAMMA, double_dqn=False))


class LearnFromMemoryTest(unittest.TestCase):
    def test_only_the_action_taken_has_an_error(self):
        np.random.seed(0)
        random.seed(0)
        model = FeedForward()
        model.add_layer(num_inputs=4, num_neurons=8, act_fn="relu")
        model.add_layer(num_inputs=8, num_neurons=2, act_fn="linear")
        model.configure(loss="mse")
        target_model = clone_model(model)

        memory = deque((np.random.normal(0, 1, (1, 4)), i % 2, 1.0, np.random.normal(0, 1, (1, 4)), False) for i in range(20))
        steps = []
        train_step = model.train_step

        # Before each step, compare its targets with what the model predicts at that moment
        def checked_train_step(inputs, targets, *args):
            steps.append((targets.copy(), model.forward_layers(inputs).copy()))
            return train_step(inputs, targets, *args)

        model.train_step = checked_train_step
        learn_from_memory(memory, model, target_model, batch_size=10, gamma=0.9, learning_rate=0.1, l1_lambda=0.0, l2_lambda=0.0, double_dqn=False)

        self.assertEqual(len(steps), 10)  # one step per sampled transition

        for targets, predictions in steps:
            differences = np.flatnonzero(targets[0] != predictions[0])
            self.assertLessEqual(len(differences), 1)  # only the action taken (unless its prediction is already right)


class ScriptedEnv:
    # Each episode lasts a scripted number of steps (reward 1 per step) and ends either by termination
    # (the pole fell) or by truncation (the time limit)
    def __init__(self, episodes):
        self.episodes = episodes  # list of (num_steps, "terminated" or "truncated")
        self.episode = -1
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(sample=lambda: 0)

    def reset(self, seed=None):
        self.episode += 1
        self.step_count = 0
        return np.full(4, float(self.episode)), {}

    def step(self, action):
        self.step_count += 1
        num_steps, ending = self.episodes[self.episode]
        done = self.step_count == num_steps
        observation = np.full(4, self.episode + self.step_count / 10)
        return observation, 1.0, done and ending == "terminated", done and ending == "truncated", {}


class DQNTrainingCase(unittest.TestCase):
    # A small linear model, a temporary directory for the files, and train() to run train_dqn on a scripted environment
    def setUp(self):
        np.random.seed(0)
        random.seed(0)

        self.model = FeedForward()
        self.model.add_layer(num_inputs=4, num_neurons=2, act_fn="linear")
        self.model.configure(loss="mse")
        self.target_model = clone_model(self.model)

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def train(self, env, **options):
        options = {"batch_size": 2, "gamma": 0.9, "learning_rate": 0.1, **options}
        metrics_filepath = os.path.join(self.directory.name, "metrics.csv")

        with mock.patch("dqn.replay_experience", wraps=replay_experience) as replay, redirect_stdout(io.StringIO()):
            rewards = train_dqn(env, self.model, self.target_model, episodes=len(env.episodes), metrics_filepath=metrics_filepath, **options)

        with open(metrics_filepath) as file:
            metrics = list(csv.DictReader(file))

        return rewards, replay, metrics


class TrainDQNTest(DQNTrainingCase):
    def test_stores_terminated_but_not_truncated_as_terminal(self):
        rewards, replay, _ = self.train(ScriptedEnv([(3, "truncated"), (2, "terminated")]), batch_size=100)

        memory = replay.call_args.args[0]  # the replay memory after the last episode
        self.assertEqual([transition[4] for transition in memory], [False, False, False, False, True])
        self.assertEqual(rewards, [3.0, 2.0])

    def test_passes_the_algorithm_to_the_replay(self):
        for double_dqn in (False, True):
            with self.subTest(double_dqn=double_dqn):
                _, replay, _ = self.train(ScriptedEnv([(3, "terminated")] * 2), double_dqn=double_dqn)
                self.assertEqual(replay.call_args.args[5], double_dqn)

    def test_target_model_is_synced_every_target_update_freq_episodes(self):
        sync = mock.Mock(wraps=self.target_model.set_layers)
        self.target_model.set_layers = sync

        # 7 episodes with target_update_freq=3: synced after episodes 1, 4 and 7, so it ends equal to the model
        self.train(ScriptedEnv([(3, "terminated")] * 7), target_update_freq=3)
        self.assertEqual(sync.call_count, 3)
        np.testing.assert_array_equal(self.target_model.layers[0]["weights"], self.model.layers[0]["weights"])

    def test_target_model_lags_between_syncs(self):
        # 6 episodes with target_update_freq=3: last synced after episode 4, and the model trained after it
        self.train(ScriptedEnv([(3, "terminated")] * 6), target_update_freq=3)
        self.assertFalse(np.array_equal(self.target_model.layers[0]["weights"], self.model.layers[0]["weights"]))

    def test_epsilon_and_learning_rate_schedules(self):
        _, _, metrics = self.train(
            ScriptedEnv([(2, "terminated")] * 5),
            epsilon=1.0, epsilon_min=0.2, epsilon_decay=0.5,
            learning_rate=0.1, learning_rate_decay=0.5, learning_rate_decay_freq=2,
        )

        # Values used in each episode; the decays apply after it (the learning rate after episodes 1, 3 and 5)
        np.testing.assert_allclose([float(row["epsilon"]) for row in metrics], [1.0, 0.5, 0.25, 0.2, 0.2])
        np.testing.assert_allclose([float(row["learning_rate"]) for row in metrics], [0.1, 0.05, 0.05, 0.025, 0.025])

    def test_checkpoint_needs_greedy_evaluations(self):
        checkpoint = os.path.join(self.directory.name, "checkpoint.joblib")

        with self.assertRaises(ValueError):
            self.train(ScriptedEnv([(2, "terminated")]), checkpoint_filepath=checkpoint)


class CountingEnv:
    # Each step gives a reward of 1. Taking action 0 ends the episode (terminated), otherwise it lasts 10 steps
    # (truncated). Records the seed of each reset.
    observation_space = SimpleNamespace(shape=(4,))

    def __init__(self):
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.step_count = 0
        return np.zeros(4), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(4), 1.0, action == 0, self.step_count == 10, {}


class EvaluateGreedyTest(unittest.TestCase):
    def test_takes_the_action_with_the_highest_q_value(self):
        prefers_1 = TableModel({0: [0.0, 1.0]})  # never takes action 0, so every episode lasts 10 steps
        prefers_0 = TableModel({0: [1.0, 0.0]})  # takes action 0 at the first step

        self.assertEqual(evaluate_greedy(CountingEnv(), prefers_1, episodes=3), 10.0)
        self.assertEqual(evaluate_greedy(CountingEnv(), prefers_0, episodes=3), 1.0)

    def test_every_evaluation_uses_the_same_starting_states(self):
        env = CountingEnv()
        evaluate_greedy(env, TableModel({0: [0.0, 1.0]}), episodes=3, seed=7)
        evaluate_greedy(env, TableModel({0: [0.0, 1.0]}), episodes=3, seed=7)

        self.assertEqual(env.seeds, [7, 8, 9, 7, 8, 9])


class TrainDQNMetricsTest(DQNTrainingCase):
    def test_greedy_evaluations_choose_the_checkpoint(self):
        save = mock.Mock(wraps=self.model.save)
        self.model.save = save
        checkpoint = os.path.join(self.directory.name, "checkpoint.joblib")

        # One evaluation episode before the training and after episodes 2, 4 and 6, with rewards 2, 5, 3 and 7
        eval_env = ScriptedEnv([(2, "terminated"), (5, "terminated"), (3, "truncated"), (7, "terminated")])
        _, _, metrics = self.train(ScriptedEnv([(3, "terminated")] * 6), eval_env=eval_env, eval_freq=2, eval_episodes=1, checkpoint_filepath=checkpoint)

        self.assertEqual([row["greedy_reward"] for row in metrics], ["", "5.0", "", "3.0", "", "7.0"])

        # Saved for the initial model (2) and when the greedy reward improved (5 and 7), not for 3
        self.assertEqual(save.call_args_list, [mock.call(checkpoint)] * 3)

    def test_mean_q_is_logged_when_the_model_trains(self):
        # The memory needs more than batch_size=2 transitions, so the training starts in the third episode
        _, _, metrics = self.train(ScriptedEnv([(1, "terminated")] * 4), batch_size=2)

        self.assertEqual([row["loss"] != "" for row in metrics], [False, False, True, True])
        self.assertEqual([row["mean_q"] != "" for row in metrics], [False, False, True, True])

    def test_metrics_file_with_other_columns_is_not_mixed(self):
        with open(os.path.join(self.directory.name, "metrics.csv"), "w") as file:
            file.write("metrics_key,episode,total_reward\n20240101000000,1,10.0\n")

        with self.assertRaises(ValueError):
            self.train(ScriptedEnv([(1, "terminated")]))


if __name__ == "__main__":
    unittest.main()
