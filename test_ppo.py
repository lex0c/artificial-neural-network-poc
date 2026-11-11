import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from activations import softmax
from feedforward import FeedForward
from ppo import collect_rollout, compute_advantages, ppo_output_delta, train_ppo
from rl import evaluate_greedy


def ppo_loss(z, actions, advantages, old_action_probs, clip_range, entropy_coef):
    # The PPO loss written directly from its definition, as a function of z of the softmax output layer
    probs = softmax(z)
    ratio = probs[np.arange(len(actions)), actions] / old_action_probs
    objective = np.minimum(ratio * advantages, np.clip(ratio, 1 - clip_range, 1 + clip_range) * advantages)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return -np.mean(objective + entropy_coef * entropy)


class PPOGradientTest(unittest.TestCase):
    def test_output_delta_matches_numerical_gradient(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, (40, 3))
        actions = rng.integers(0, 3, 40)
        advantages = rng.normal(0, 1, 40)  # positive and negative

        # Old probabilities around the current ones, so some ratios fall outside [0.8, 1.2] and get clipped
        current = softmax(z)[np.arange(40), actions]
        old_action_probs = current / rng.uniform(0.6, 1.4, 40)

        delta, stats = ppo_output_delta(softmax(z), actions, advantages, old_action_probs, clip_range=0.2, entropy_coef=0.01)
        self.assertGreater(stats["clip_fraction"], 0)
        self.assertLess(stats["clip_fraction"], 1)

        eps = 1e-6
        numerical = np.zeros_like(z)
        for idx in np.ndindex(z.shape):
            plus, minus = z.copy(), z.copy()
            plus[idx] += eps
            minus[idx] -= eps
            numerical[idx] = (ppo_loss(plus, actions, advantages, old_action_probs, 0.2, 0.01) - ppo_loss(minus, actions, advantages, old_action_probs, 0.2, 0.01)) / (2 * eps)

        np.testing.assert_allclose(delta, numerical, rtol=1e-5, atol=1e-9)
        self.assertAlmostEqual(stats["loss"], ppo_loss(z, actions, advantages, old_action_probs, 0.2, 0.01))


class AdvantagesTest(unittest.TestCase):
    # 3 steps with reward 1, every value 0.5, gamma 0.5 and lambda 0.5 (so each later advantage counts 0.25)
    REWARDS = np.ones(3)
    VALUES = np.full(3, 0.5)

    def advantages(self, next_values, terminated, truncated):
        return compute_advantages(self.REWARDS, self.VALUES, np.array(next_values), np.array(terminated), np.array(truncated), gamma=0.5, gae_lambda=0.5)

    def test_episode_that_terminates(self):
        # delta = 1 + 0.5 * 0.5 - 0.5 = 0.75, except the last one: 1 - 0.5 = 0.5 (no future after termination)
        # advantage[2] = 0.5, advantage[1] = 0.75 + 0.25 * 0.5 = 0.875, advantage[0] = 0.75 + 0.25 * 0.875 = 0.96875
        advantages, returns = self.advantages([0.5, 0.5, 0.5], [False, False, True], [False, False, False])

        np.testing.assert_allclose(advantages, [0.96875, 0.875, 0.5])
        np.testing.assert_allclose(returns, advantages + 0.5)

    def test_truncated_episode_bootstraps_from_its_last_state(self):
        # The last state is worth 2: delta[2] = 1 + 0.5 * 2 - 0.5 = 1.5, then 0.75 + 0.25 * 1.5 = 1.125 and 1.03125
        advantages, _ = self.advantages([0.5, 0.5, 2.0], [False, False, False], [False, False, True])

        np.testing.assert_allclose(advantages, [1.03125, 1.125, 1.5])

    def test_advantages_dont_cross_episodes(self):
        # The first episode ends at step 1: advantage[1] = 1 - 0.5 = 0.5 doesn't include the next episode
        advantages, _ = self.advantages([0.5, 0.5, 0.5], [False, True, False], [False, False, False])

        np.testing.assert_allclose(advantages, [0.75 + 0.25 * 0.5, 0.5, 0.75])


class CountingEnv:
    # Episodes of 3 steps, reward 1 per step; the state is [episode number, step]
    observation_space = SimpleNamespace(shape=(2,))
    action_space = SimpleNamespace(n=2)

    def __init__(self):
        self.episode = -1

    def reset(self, seed=None):
        self.episode += 1
        self.step_count = 0
        return np.array([self.episode, 0.0]), {}

    def step(self, action):
        self.step_count += 1
        return np.array([self.episode, self.step_count]), 1.0, False, self.step_count == 3, {}


class UniformPolicy:
    def predict(self, states):
        return np.full((len(states), 2), 0.5)


class CollectRolloutTest(unittest.TestCase):
    def test_episodes_continue_between_calls(self):
        np.random.seed(0)
        env = CountingEnv()

        rollout, state, episode_reward, finished = collect_rollout(env, UniformPolicy(), num_steps=4)
        self.assertEqual(finished, [3.0])  # the first episode ended after 3 steps
        self.assertEqual(episode_reward, 1.0)  # one step of the second episode
        self.assertEqual(list(rollout["truncated"]), [False, False, True, False])
        np.testing.assert_array_equal(rollout["next_states"][2], [0, 3])  # the last state of the episode, not the reset one

        rollout, _, episode_reward, finished = collect_rollout(env, UniformPolicy(), 2, state, episode_reward)
        self.assertEqual(finished, [3.0])  # the second episode, counting the step of the previous call
        np.testing.assert_array_equal(rollout["states"][0], [1, 1])


class TwoArmedBandit:
    # One-step episodes: action 1 gives a reward of 1, action 0 nothing
    observation_space = SimpleNamespace(shape=(1,))
    action_space = SimpleNamespace(n=2)

    def reset(self, seed=None):
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), float(action == 1), True, False, {}


def bandit_models():
    np.random.seed(0)
    policy = FeedForward()
    policy.add_layer(num_inputs=1, num_neurons=2, act_fn="softmax")
    policy.configure(loss=None, optimizer="adam")

    value_model = FeedForward()
    value_model.add_layer(num_inputs=1, num_neurons=1, act_fn="linear")
    value_model.configure(loss="mse", optimizer="adam")
    return policy, value_model


# The size of each minibatch of one iteration with one epoch: the rows of the probabilities given to ppo_output_delta
def minibatch_sizes(num_steps, minibatch_size):
    policy, value_model = bandit_models()

    with mock.patch("ppo.ppo_output_delta", wraps=ppo_output_delta) as delta, redirect_stdout(io.StringIO()):
        train_ppo(TwoArmedBandit(), policy, value_model, total_steps=num_steps, num_steps=num_steps,
                  minibatch_size=minibatch_size, num_epochs=1)

    return [call.args[0].shape[0] for call in delta.call_args_list]


# The steps of each rollout, and how many greedy evaluations ran (counting the one before training)
def rollouts_and_evaluations(total_steps, num_steps, eval_freq):
    policy, value_model = bandit_models()

    with mock.patch("ppo.collect_rollout", wraps=collect_rollout) as collect, \
         mock.patch("ppo.evaluate_greedy", wraps=evaluate_greedy) as evaluate, redirect_stdout(io.StringIO()):
        train_ppo(TwoArmedBandit(), policy, value_model, total_steps=total_steps, num_steps=num_steps, minibatch_size=16,
                  num_epochs=1, eval_env=TwoArmedBandit(), eval_freq=eval_freq, eval_episodes=1)

    return [call.args[2] for call in collect.call_args_list], evaluate.call_count


class TrainPPOTest(unittest.TestCase):
    def test_learns_the_better_action(self):
        policy, value_model = bandit_models()

        with redirect_stdout(io.StringIO()):
            train_ppo(TwoArmedBandit(), policy, value_model, total_steps=1280, learning_rate=0.01, num_steps=64, minibatch_size=16, num_epochs=4)

        self.assertGreater(policy.predict(np.zeros((1, 1)))[0, 1], 0.95)

    def test_minibatches_have_at_most_minibatch_size_steps(self):
        self.assertEqual(minibatch_sizes(128, 64), [64, 64])
        self.assertEqual(minibatch_sizes(100, 64), [64, 36])  # the last one gets the remaining steps

    def test_rollout_shorter_than_a_minibatch_is_a_single_minibatch(self):
        self.assertEqual(minibatch_sizes(32, 64), [32])

    def test_budget_shorter_than_a_rollout_trains_and_evaluates_it(self):
        # Before, no iteration ran, and the checkpoint kept the untrained policy
        self.assertEqual(rollouts_and_evaluations(total_steps=32, num_steps=64, eval_freq=5), ([32], 2))

    def test_last_rollout_plays_the_steps_left(self):
        rollouts, _ = rollouts_and_evaluations(total_steps=100, num_steps=64, eval_freq=5)
        self.assertEqual(rollouts, [64, 36])

    def test_longer_training_is_evaluated_every_eval_freq_iterations(self):
        # Before training and after iteration 5, but not at the end (iteration 6)
        self.assertEqual(rollouts_and_evaluations(total_steps=6 * 64, num_steps=64, eval_freq=5), ([64] * 6, 2))

    def test_rejects_a_budget_without_steps_before_saving_anything(self):
        policy, value_model = bandit_models()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.joblib")
            with self.assertRaises(ValueError):
                train_ppo(TwoArmedBandit(), policy, value_model, total_steps=0, eval_env=TwoArmedBandit(),
                          checkpoint_filepath=checkpoint)
            self.assertFalse(os.path.exists(checkpoint))


if __name__ == "__main__":
    unittest.main()
