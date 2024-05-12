import numpy as np
import gymnasium as gym
from datetime import datetime
from collections import deque
import random
import time
import csv

from feedforward import load_model


#model = load_model('models/cartpole.joblib')
model = load_model('models/cartpole-finetuned.joblib')
model.summary()

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

episodes = 500
memory = deque(maxlen=100000)
batch_size = 512
train_data = {'states': [], 'targets': []}

epsilon = 0.1
gamma = 0.95  # high values cause the gradient to explode
learning_rate = 0.0001
learning_rate_decay = 0.99

rolling_rewards = deque(maxlen=100)

metrics_filepath = 'metrics/cartpole.csv'
metrics_key = datetime.now().strftime('%Y%m%d%H%M%S')


def write_to_csv(metric):
    with open(metrics_filepath, 'a', newline='') as file:
        fieldnames = list(metric.keys())

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(metric)


def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()

    q_values = model.predict(state)

    return np.argmax(q_values)


def replay_experience(replay_memory):
    if len(replay_memory) <= batch_size:
        return # Ensure there is enough data in memory to sample a batch

    # Sample a random minibatch of transitions (state, action, reward, next_state, done) from the memory
    minibatch = random.sample(replay_memory, batch_size)

    # Extract elements of each experience
    states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

    # Reshape states and next_states for prediction
    states = np.vstack(states)  # Stack states vertically to create a proper batch
    next_states = np.vstack(next_states)  # Stack next states vertically

    # Predict Q-values for next_states and current states in one go
    next_q_values = model.predict(next_states)
    current_q_values = model.predict(states)

    # Calculate the maximum Q-value for each next state
    max_next_q_values = np.max(next_q_values, axis=1)

    # Compute target Q-values using the Bellman equation
    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Update the Q-values for the actions taken
    targets = current_q_values.copy()  # Start with current predictions
    for idx, action in enumerate(actions):
        targets[idx, action] = target_q_values[idx]  # Update only the actions that were taken

    return states, targets


def train_model(train_data):
    if train_data['states']:
        loss, _ = model.train(np.array(train_data['states']), np.array(train_data['targets']), epochs=1, learning_rate=learning_rate, l1_lambda=0.0001, l2_lambda=0.0001)
        train_data['states'].clear()
        train_data['targets'].clear()
        return loss


for e in range(episodes):
    start_time = time.time()

    state, _ = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False
    total_reward = 0
    step_count = 0

    while True:
        action = choose_action(state)

        observation, reward, terminated, truncated, info = env.step(action)
        next_state = np.reshape(observation, [1, env.observation_space.shape[0]])
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
        step_count += 1

        if done:
            end_time = time.time()
            episode_duration = end_time - start_time

            rolling_rewards.append(total_reward)
            average_rolling_reward = sum(rolling_rewards) / len(rolling_rewards)

            training_data = replay_experience(memory)

            if training_data:
                states, targets = training_data
                train_data['states'].extend(states)
                train_data['targets'].extend(targets)

            loss = None
            if len(train_data['states']) >= batch_size:
                loss = train_model(train_data)

            write_to_csv({
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
            break

    if e % 300 == 0:
        learning_rate *= learning_rate_decay


model.save('models/cartpole-finetuned.joblib')
env.close()

