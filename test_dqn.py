import random
import unittest
from collections import deque

import numpy as np

from dqn import replay_experience


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
    # Sampling with several seeds checks all of them; the rows are identified by their state.
    found = {}

    for seed in range(20):
        random.seed(seed)
        memory = deque(TRANSITIONS.values())
        states, targets = replay_experience(memory, ONLINE, TARGET, batch_size=len(memory) - 1, gamma=GAMMA, double_dqn=double_dqn)
        found.update({int(s[0]): t for s, t in zip(states, targets)})

    return found


class ReplayExperienceTest(unittest.TestCase):
    def assert_targets(self, double_dqn, next_value_of_state_10):
        targets = targets_by_state(double_dqn)
        self.assertEqual(set(targets), set(TRANSITIONS))

        for state_id, target in targets.items():
            _, action, reward, _, terminal = TRANSITIONS[state_id]
            expected = ONLINE.table[state_id].copy()  # actions not taken keep the current prediction
            expected[action] = reward if terminal else reward + GAMMA * next_value_of_state_10

            np.testing.assert_allclose(target, expected, err_msg=f"state {state_id}")

    def test_dqn_uses_the_max_of_the_target_model(self):
        self.assert_targets(double_dqn=False, next_value_of_state_10=5.0)

    def test_double_dqn_evaluates_the_online_choice_with_the_target_model(self):
        # Online argmax in state 10 is action 1, whose target value is 2.0
        self.assert_targets(double_dqn=True, next_value_of_state_10=2.0)

    def test_needs_more_transitions_than_the_batch_size(self):
        memory = deque(TRANSITIONS.values())
        self.assertIsNone(replay_experience(memory, ONLINE, TARGET, batch_size=len(memory), gamma=GAMMA, double_dqn=False))


if __name__ == "__main__":
    unittest.main()
