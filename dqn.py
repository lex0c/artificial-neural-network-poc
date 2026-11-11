import csv
import random
import time
from collections import deque
from datetime import datetime

import numpy as np


# Deep Q-Network (DQN): an agent learns to act in an environment by estimating Q(state, action), the total
# discounted reward it can expect after taking that action in that state and acting well afterwards. The best action
# in a state is the one with the highest Q-value. Here the Q-values come from a FeedForward model with one output per
# action. DQN.md explains the concepts and how this training loop works.


# Epsilon-greedy policy: with probability epsilon a random action (exploration: trying actions the model doesn't
# rank first yet), otherwise the action with the highest Q-value (exploitation).
def choose_action(env, model, state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()

    q_values = model.predict(state)

    return np.argmax(q_values)


# Plays one episode with the epsilon-greedy policy and stores every transition in the replay memory.
# Returns the total reward and the number of steps.
def play_episode(env, model, memory, epsilon):
    num_observations = env.observation_space.shape[0]

    state, _ = env.reset()
    state = np.reshape(state, [1, num_observations])
    total_reward = 0
    steps = 0

    while True:
        action = choose_action(env, model, state, epsilon)

        observation, reward, terminated, truncated, info = env.step(action)
        next_state = np.reshape(observation, [1, num_observations])

        # terminated: the episode really ended (no future rewards). truncated: it was cut by the time limit, and the
        # future would have gone on, so it isn't stored as terminal.
        memory.append((state, action, reward, next_state, terminated))

        state = next_state
        total_reward += reward
        steps += 1

        if terminated or truncated:
            return total_reward, steps


# Builds a training batch from random transitions of the replay memory: the states, and the Q-values the model should
# predict for them, the Bellman targets:
#   target = reward + gamma * max over the actions a' of Q_target(next_state, a')
#   target = reward, if the episode ended in next_state
# Only the Q-value of the action that was taken gets a Bellman target; the others get their current prediction, so at
# first their error is zero. learn_from_memory then makes one update per transition, and each update also changes the
# predictions of the other transitions: by then those targets are slightly outdated, and they pull the Q-values of the
# actions not taken back towards their values before the updates (see DQN.md).
#
# Sampling at random, instead of learning from the last steps in order, breaks the correlation between consecutive
# states (they are almost the same), and reuses each experience many times.
def replay_experience(memory, model, target_model, batch_size, gamma, double_dqn):
    if len(memory) <= batch_size:
        return None  # not enough transitions yet

    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, terminals = map(np.array, zip(*minibatch))

    states = np.vstack(states)  # shape (batch_size, num_observations)
    next_states = np.vstack(next_states)

    next_q_values = target_model.predict(next_states)
    current_q_values = model.predict(states)

    if double_dqn:
        # Double DQN: the model chooses the best next action and the target model evaluates it. Taking the max over
        # the same noisy estimates that choose the action tends to overestimate the Q-values.
        next_actions = np.argmax(model.predict(next_states), axis=1)
        next_values = next_q_values[np.arange(batch_size), next_actions]
    else:
        # DQN: the target model both chooses and evaluates the best next action
        next_values = np.max(next_q_values, axis=1)

    target_q_values = rewards + gamma * next_values * (1 - terminals)

    targets = current_q_values.copy()
    targets[np.arange(batch_size), actions] = target_q_values

    return states, targets


# Fits the model to the Bellman targets of a random batch of transitions from the replay memory.
#
# This project learns once per episode, with one update per sampled transition (batch_size=1 in model.train). The
# original DQN instead does one minibatch update after every step of the environment.
#
# Returns the loss and the mean of the highest Q-value of the sampled states (None while the memory is too small).
# The mean Q-value should level off near the discounted return the agent gets, at most 1 / (1 - gamma) with a reward
# of 1 per step; growing beyond it means the Q-values are overestimated.
def learn_from_memory(memory, model, target_model, batch_size, gamma, learning_rate, l1_lambda, l2_lambda, double_dqn):
    training_data = replay_experience(memory, model, target_model, batch_size, gamma, double_dqn)

    if training_data is None:
        return None, None

    states, targets = training_data
    mean_q = float(np.mean(np.max(model.predict(states), axis=1)))

    # shuffle=False: the batch is already a random sample of the memory
    history = model.train(states, targets, epochs=1, learning_rate=learning_rate, batch_size=1, l1_lambda=l1_lambda, l2_lambda=l2_lambda, shuffle=False)

    return history["loss"][0], mean_q


# Plays episodes with the greedy policy (always the action with the highest Q-value, no exploration) and returns the
# mean total reward: what the agent has learned, without the random actions of the training. Each episode starts from
# a fixed seed, so every evaluation along the training uses the same starting states.
def evaluate_greedy(env, model, episodes=10, seed=0):
    num_observations = env.observation_space.shape[0]
    total_rewards = []

    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        total_reward = 0

        while True:
            q_values = model.predict(np.reshape(state, [1, num_observations]))
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q_values)))
            total_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(total_reward)

    return float(np.mean(total_rewards))


def write_to_csv(filepath, metric):
    fieldnames = list(metric.keys())

    with open(filepath, 'a+', newline='') as file:
        file.seek(0)
        header = file.readline().strip()

        # Rows with other columns under the existing header would misalign the whole file
        if header and header != ','.join(fieldnames):
            raise ValueError(f"{filepath} has other columns ({header}); move it away to start a new metrics file")

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not header:
            writer.writeheader()

        writer.writerow(metric)


# Trains a DQN agent, updating the model in place. Each episode:
#   1. act: play an episode with the epsilon-greedy policy, storing the transitions in the replay memory
#   2. learn: fit the model to the Bellman targets of a random batch from the memory
#   3. evaluate: every eval_freq episodes, play eval_episodes with the greedy policy in eval_env (a separate
#      environment, so the training one isn't disturbed), and save the checkpoint when it is the best so far
#   4. schedules: explore less (epsilon) and, optionally, lower the learning rate
#   5. every target_update_freq episodes, copy the model into the target model
#
# The checkpoint keeps the model with the best greedy evaluation, starting with the initial model (evaluated before the
# training): a DQN policy often gets worse after reaching its best, so the last model isn't necessarily the best one.
#
# Returns the total reward of each episode.
def train_dqn(env, model, target_model, episodes, batch_size, gamma, learning_rate,
              epsilon=1.0, epsilon_min=0.1, epsilon_decay=1.0,
              learning_rate_decay=1.0, learning_rate_decay_freq=300,
              l1_lambda=0.0, l2_lambda=0.0,
              target_update_freq=10, double_dqn=False, memory_size=100000,
              eval_env=None, eval_freq=50, eval_episodes=10,
              checkpoint_filepath=None, metrics_filepath='metrics/cartpole.csv'):
    if checkpoint_filepath and eval_env is None:
        raise ValueError("checkpoint_filepath needs eval_env: the checkpoint is the model with the best greedy evaluation")

    memory = deque(maxlen=memory_size)  # the oldest transitions are dropped when it is full
    rolling_rewards = deque(maxlen=100)
    episode_rewards = []
    metrics_key = datetime.now().strftime('%Y%m%d%H%M%S')

    if eval_env is not None:
        best_greedy_reward = evaluate_greedy(eval_env, model, eval_episodes)
        print(f"greedy evaluation before training: {best_greedy_reward:.2f}")

        if checkpoint_filepath:
            model.save(checkpoint_filepath)

    for e in range(episodes):
        start_time = time.time()

        # 1. Act
        total_reward, steps = play_episode(env, model, memory, epsilon)
        episode_duration = time.time() - start_time

        episode_rewards.append(total_reward)
        rolling_rewards.append(total_reward)
        average_rolling_reward = sum(rolling_rewards) / len(rolling_rewards)

        # 2. Learn
        loss, mean_q = learn_from_memory(memory, model, target_model, batch_size, gamma, learning_rate, l1_lambda, l2_lambda, double_dqn)

        # 3. Evaluate
        greedy_reward = None

        if eval_env is not None and (e + 1) % eval_freq == 0:
            greedy_reward = evaluate_greedy(eval_env, model, eval_episodes)

            if greedy_reward > best_greedy_reward:
                best_greedy_reward = greedy_reward

                if checkpoint_filepath:
                    model.save(checkpoint_filepath)

        write_to_csv(metrics_filepath, {
            'metrics_key': metrics_key,
            'episode': e+1,
            'total_reward': total_reward,
            'average_rolling_reward': average_rolling_reward,
            'greedy_reward': greedy_reward,
            'mean_q': mean_q,
            'loss': loss,
            'duration': episode_duration,
            'epsilon': epsilon,
            'steps': steps,
            'learning_rate': learning_rate
        })

        print(f"episode: {e+1}/{episodes} - reward: {total_reward:.2f} - duration: {episode_duration:.2f}s - epsilon: {epsilon:.3f} - steps: {steps}")

        if greedy_reward is not None:
            print(f"greedy evaluation: {greedy_reward:.2f} (best: {best_greedy_reward:.2f})")

        # 4. Schedules
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if e % learning_rate_decay_freq == 0:
            learning_rate *= learning_rate_decay

        # 5. Target model
        if e % target_update_freq == 0:
            target_model.set_layers(model.get_layers())

    return episode_rewards
