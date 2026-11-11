import csv
import random
import time
from collections import deque
from datetime import datetime

import numpy as np


def write_to_csv(filepath, metric):
    with open(filepath, 'a', newline='') as file:
        fieldnames = list(metric.keys())

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(metric)


# Epsilon-greedy policy: explores with a random action with probability epsilon, otherwise takes the best known action.
def choose_action(env, model, state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()

    q_values = model.predict(state)

    return np.argmax(q_values)


# Builds a training batch (states and their target Q-values) from random transitions of the replay memory.
def replay_experience(memory, model, target_model, batch_size, gamma, double_dqn):
    if len(memory) <= batch_size:
        return None  # Ensure there is enough data in memory to sample a batch

    # Sample a random minibatch of transitions (state, action, reward, next_state, terminal) from the memory
    minibatch = random.sample(memory, batch_size)

    # Extract elements of each experience
    states, actions, rewards, next_states, terminals = map(np.array, zip(*minibatch))

    # Stack states vertically to create a proper batch
    states = np.vstack(states)
    next_states = np.vstack(next_states)

    next_q_values = target_model.predict(next_states)
    current_q_values = model.predict(states)

    if double_dqn:
        # Double DQN: the online model chooses the best next action and the target model evaluates it.
        # Taking the max over the same noisy estimates that choose the action overestimates the Q-values.
        next_actions = np.argmax(model.predict(next_states), axis=1)
        next_values = next_q_values[np.arange(batch_size), next_actions]
    else:
        # DQN: the target model both chooses and evaluates the best next action
        next_values = np.max(next_q_values, axis=1)

    # Compute target Q-values using the Bellman equation.
    # Only a real terminal state (pole fell or cart left the track) has no future value; hitting the time limit
    # (truncation) still bootstraps
    target_q_values = rewards + gamma * next_values * (1 - terminals)

    # Update only the Q-values of the actions that were taken, the others keep the current predictions
    targets = current_q_values.copy()
    targets[np.arange(batch_size), actions] = target_q_values

    return states, targets


# Trains a Deep Q-Network agent, updating the model in place.
#
# After each episode, a batch is sampled from the replay memory and the model is trained on it with one update per
# transition. The target model is a copy of the model that is refreshed every target_update_freq episodes.
#
# Returns the total reward of each episode.
def train_dqn(env, model, target_model, episodes, batch_size, gamma, learning_rate,
              epsilon=1.0, epsilon_min=0.1, epsilon_decay=1.0,
              learning_rate_decay=1.0, learning_rate_decay_freq=300,
              l1_lambda=0.0, l2_lambda=0.0,
              target_update_freq=10, double_dqn=False, memory_size=100000,
              checkpoint_filepath=None, metrics_filepath='metrics/cartpole.csv'):
    memory = deque(maxlen=memory_size)
    rolling_rewards = deque(maxlen=100)
    best_average_rolling_reward = 0
    episode_rewards = []

    metrics_key = datetime.now().strftime('%Y%m%d%H%M%S')
    num_observations = env.observation_space.shape[0]

    for e in range(episodes):
        start_time = time.time()

        state, _ = env.reset()
        state = np.reshape(state, [1, num_observations])
        total_reward = 0
        step_count = 0

        while True:
            action = choose_action(env, model, state, epsilon)

            observation, reward, terminated, truncated, info = env.step(action)
            next_state = np.reshape(observation, [1, num_observations])

            memory.append((state, action, reward, next_state, terminated))

            state = next_state
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        episode_duration = time.time() - start_time
        episode_rewards.append(total_reward)

        rolling_rewards.append(total_reward)
        average_rolling_reward = sum(rolling_rewards) / len(rolling_rewards)

        # Keeps the model with the best average rolling reward so far
        if checkpoint_filepath and average_rolling_reward > best_average_rolling_reward:
            best_average_rolling_reward = average_rolling_reward
            model.save(checkpoint_filepath)

        loss = None
        training_data = replay_experience(memory, model, target_model, batch_size, gamma, double_dqn)

        if training_data:
            states, targets = training_data
            loss, _ = model.train(states, targets, epochs=1, learning_rate=learning_rate, batch_size=1, l1_lambda=l1_lambda, l2_lambda=l2_lambda)  # one update per transition

        write_to_csv(metrics_filepath, {
            'metrics_key': metrics_key,
            'episode': e+1,
            'total_reward': total_reward,
            'average_rolling_reward': average_rolling_reward,
            'loss': loss,
            'duration': episode_duration,
            'epsilon': epsilon,
            'steps': step_count,
            'learning_rate': learning_rate
        })

        print(f"episode: {e+1}/{episodes} - reward: {total_reward:.2f} - duration: {episode_duration:.2f}s - epsilon: {epsilon:.3f} - steps: {step_count}")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if e % learning_rate_decay_freq == 0:
            learning_rate *= learning_rate_decay

        if e % target_update_freq == 0:
            target_model.set_layers(model.get_layers())  # Update target network

    return episode_rewards
