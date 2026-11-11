import random
import time
from collections import deque
from datetime import datetime

import numpy as np

from progress import display_path, format_duration, format_number, timing
from rl import evaluate_greedy, write_to_csv


# Deep Q-Network (DQN): an agent learns to act in an environment by estimating Q(state, action), the total
# discounted reward it can expect after taking that action in that state and acting well afterwards. The best action
# in a state is the one with the highest Q-value. Here the Q-values come from a FeedForward model with one output per
# action. DQN.md explains the concepts and how this training loop works.


# How many steps a random action is held: 1 for classic epsilon-greedy, otherwise a random duration from 1 to
# max_repeat where short durations are the most likely (a zeta distribution, as in Dabney et al. 2020).
def repeat_duration(max_repeat):
    if max_repeat == 1:
        return 1

    return int(min(np.random.zipf(2.0), max_repeat))


# Plays one episode with the epsilon-greedy policy and stores every transition in the replay memory:
# with probability epsilon a random action (exploration: trying actions the model doesn't rank first yet), otherwise
# the action with the highest Q-value (exploitation).
#
# With max_repeat > 1, each random action is held for a random number of steps (temporally extended exploration, also
# called ez-greedy). Changing to a random action at every step mostly cancels out, while holding an action builds
# momentum, which environments such as Acrobot need to ever reach their goal by chance.
#
# Returns the total reward and the number of steps.
def play_episode(env, model, memory, epsilon, max_repeat=1):
    num_observations = env.observation_space.shape[0]

    state, _ = env.reset()
    state = np.reshape(state, [1, num_observations])
    total_reward = 0
    steps = 0
    random_steps_left = 0

    while True:
        if random_steps_left > 0:  # still holding a random action
            random_steps_left -= 1
        elif np.random.rand() < epsilon:  # explore
            action = env.action_space.sample()
            random_steps_left = repeat_duration(max_repeat) - 1
        else:  # exploit
            action = np.argmax(model.predict(state))

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


# Samples a random batch of transitions from the replay memory and computes their Bellman targets, the Q-value each
# action taken should have:
#   target = reward + gamma * max over the actions a' of Q_target(next_state, a')
#   target = reward, if the episode ended in next_state
# Returns the states, the actions taken and their targets (None while the memory isn't larger than batch_size).
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

    if double_dqn:
        # Double DQN: the model chooses the best next action and the target model evaluates it. Taking the max over
        # the same noisy estimates that choose the action tends to overestimate the Q-values.
        next_actions = np.argmax(model.predict(next_states), axis=1)
        next_values = next_q_values[np.arange(batch_size), next_actions]
    else:
        # DQN: the target model both chooses and evaluates the best next action
        next_values = np.max(next_q_values, axis=1)

    target_q_values = rewards + gamma * next_values * (1 - terminals)

    return states, actions, target_q_values


# Fits the model to the Bellman targets of a random batch of transitions from the replay memory, with one gradient
# descent step per minibatch of minibatch_size transitions (batch_size / minibatch_size steps).
#
# The model has one output per action, but a transition only says something about the action that was taken. So the
# target of each step is the current prediction of the model, made right before the step, with the Bellman target in
# place of the action taken: the other outputs have no error and only the Q-values of the actions taken are trained.
# The Bellman targets can be computed once for the whole batch, because the target model doesn't change during it.
#
# This project learns once per episode. The original DQN instead does one minibatch update after every step of the
# environment.
#
# Returns the mean loss of the steps and the mean of the highest Q-value of the sampled states (None while the memory
# is too small). The mean Q-value should level off near the discounted return the agent gets, always below
# 1 / (1 - gamma) with a reward of 1 per step; values beyond it mean the Q-values are overestimated.
def learn_from_memory(memory, model, target_model, batch_size, gamma, learning_rate, l1_lambda, l2_lambda, double_dqn, minibatch_size=1):
    training_data = replay_experience(memory, model, target_model, batch_size, gamma, double_dqn)

    if training_data is None:
        return None, None

    states, actions, target_q_values = training_data
    mean_q = float(np.mean(np.max(model.predict(states), axis=1)))
    losses = []

    for start in range(0, len(states), minibatch_size):
        minibatch = slice(start, start + minibatch_size)
        minibatch_states = model.normalize(states[minibatch])  # train_step expects normalized inputs, as train() gives

        targets = model.forward_layers(minibatch_states).copy()  # the current prediction for every action...
        targets[np.arange(len(minibatch_states)), actions[minibatch]] = target_q_values[minibatch]  # ...but the action taken

        losses.append(model.train_step(minibatch_states, targets, learning_rate, l1_lambda, l2_lambda))

    return float(np.mean(losses)), mean_q


# Trains a DQN agent, updating the model in place. Each episode:
#   1. act: play an episode with the epsilon-greedy policy (holding each random action for up to max_repeat steps),
#      storing the transitions in the replay memory
#   2. learn: fit the model to the Bellman targets of a random batch of batch_size transitions from the memory, with
#      one gradient descent step per minibatch of minibatch_size transitions
#   3. evaluate: every eval_freq episodes, play eval_episodes with the greedy policy in eval_env (a separate
#      environment, so the training one isn't disturbed), and save the checkpoint when it is the best so far
#   4. schedules: explore less (epsilon) and, optionally, lower the learning rate
#   5. every target_update_freq episodes, copy the model into the target model
#
# The checkpoint keeps the model with the best greedy evaluation, starting with the initial model (evaluated before the
# training): a DQN policy often gets worse after reaching its best, so the last model isn't necessarily the best one.
# With metrics_filepath, every episode adds a row of metrics to that CSV file (see show-metrics.py).
#
# It works with any Gymnasium environment whose actions are discrete (one Q-value per action) and whose state is a
# vector of numbers.
#
# Returns the total reward of each episode.
def train_dqn(env, model, target_model, episodes, batch_size, gamma, learning_rate,
              epsilon=1.0, epsilon_min=0.1, epsilon_decay=1.0,
              learning_rate_decay=1.0, learning_rate_decay_freq=300,
              l1_lambda=0.0, l2_lambda=0.0,
              target_update_freq=10, double_dqn=False, memory_size=100000, minibatch_size=1, max_repeat=1,
              eval_env=None, eval_freq=50, eval_episodes=10,
              checkpoint_filepath=None, metrics_filepath=None, log_every=10):
    if checkpoint_filepath and eval_env is None:
        raise ValueError("checkpoint_filepath needs eval_env: the checkpoint is the model with the best greedy evaluation")

    memory = deque(maxlen=memory_size)  # the oldest transitions are dropped when it is full
    rolling_rewards = deque(maxlen=100)
    episode_rewards = []
    metrics_key = datetime.now().strftime('%Y%m%d%H%M%S')
    training_start = time.time()

    if eval_env is not None:
        best_greedy_reward, best_episode = evaluate_greedy(eval_env, model, eval_episodes), 0
        print(f"Greedy evaluation before training: {best_greedy_reward:.1f}")

        if checkpoint_filepath:
            model.save(checkpoint_filepath)

    for e in range(episodes):
        start_time = time.time()

        # 1. Act
        total_reward, steps = play_episode(env, model, memory, epsilon, max_repeat)
        episode_duration = time.time() - start_time

        episode_rewards.append(total_reward)
        rolling_rewards.append(total_reward)
        average_rolling_reward = sum(rolling_rewards) / len(rolling_rewards)

        # 2. Learn
        loss, mean_q = learn_from_memory(memory, model, target_model, batch_size, gamma, learning_rate, l1_lambda, l2_lambda, double_dqn, minibatch_size)

        # 3. Evaluate
        greedy_reward = None

        if eval_env is not None and (e + 1) % eval_freq == 0:
            greedy_reward = evaluate_greedy(eval_env, model, eval_episodes)

            if greedy_reward > best_greedy_reward:
                best_greedy_reward, best_episode = greedy_reward, e + 1

                if checkpoint_filepath:
                    model.save(checkpoint_filepath)

        model.notify("on_metrics", {'episode': e + 1, 'total_reward': total_reward, 'greedy_reward': greedy_reward,
                                    'mean_q': mean_q, 'loss': loss, 'epsilon': epsilon})

        if metrics_filepath:
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

        # Progress: one line every log_every episodes, with the rewards of those episodes
        if (e + 1) % log_every == 0 or e + 1 == episodes:
            block = episode_rewards[-((e % log_every) + 1):]
            print(f"Episodes {e + 2 - len(block)}-{e + 1}/{episodes} - reward mean {sum(block) / len(block):.1f}, max {max(block):.0f} "
                  f"- epsilon {epsilon:.3f} - loss {format_number(loss)} - {timing(training_start, e + 1, episodes)}")

        if greedy_reward is not None:
            print(f"  Greedy evaluation: {greedy_reward:.1f} (best: {best_greedy_reward:.1f})")

        # 4. Schedules
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if e % learning_rate_decay_freq == 0:
            learning_rate *= learning_rate_decay

        # 5. Target model
        if e % target_update_freq == 0:
            target_model.set_layers(model.get_layers())

    print(f"\nDone in {format_duration(time.time() - training_start)}.")

    if eval_env is not None:
        when = f"after episode {best_episode}" if best_episode else "before training"
        print(f"Best greedy evaluation: {best_greedy_reward:.1f} ({when})")

    if checkpoint_filepath:
        print(f"Best model saved to {display_path(checkpoint_filepath)}")

    if metrics_filepath:
        print(f"Metrics saved to {display_path(metrics_filepath)} (plot them with: python show-metrics.py {display_path(metrics_filepath)})")

    return episode_rewards
