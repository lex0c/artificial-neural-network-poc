# Deep Q-Network (DQN)

`dqn.py` trains an agent to balance a pole on a cart ([CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole))
with a Deep Q-Network: a `FeedForward` model (see [FEEDFORWARD.md](FEEDFORWARD.md)) that learns the value of each
action. This page explains the ideas and how the training loop works.

## The problem

In reinforcement learning, an **agent** observes the **state** of an **environment**, takes an **action**, and gets a
**reward** and the next state. Nobody tells it the right action: it has to find out which actions lead to more reward
over time.

In CartPole:

- **State**: 4 numbers, the cart position, the cart velocity, the pole angle and the pole angular velocity.
- **Actions**: 0 pushes the cart to the left, 1 to the right.
- **Reward**: +1 for every step the pole stays up.
- **End of an episode**: it is **terminated** when the pole tilts more than 12 degrees or the cart leaves the track
  (more than 2.4 from the center), and **truncated** when it reaches the time limit of 500 steps. 500 is the best score.

## Q-values

The agent's goal is the **return**, the sum of the rewards it will get, where a reward `k` steps ahead is multiplied by
`gamma^k`:

```
return = r0 + gamma * r1 + gamma^2 * r2 + ...
```

`gamma` (the **discount factor**, between 0 and 1) sets how far ahead the agent looks. The weights `gamma^k` add up to
`1 / (1 - gamma)`, often called the horizon. With `gamma = 0.8`, as in `train-cartpole.py`, the horizon is 5 steps: a
reward 5 steps ahead still counts `0.8^5 = 0.33`, and one 20 steps ahead about 0.01. With a reward of 1 per step, the
return is always below `1 / (1 - 0.8) = 5`.

**Q(state, action)** is the return the agent can expect after taking `action` in `state` and acting well afterwards.
If it knew the Q-values, the best policy would be simple: in each state, take the action with the highest Q-value.

The **Q-network** estimates them: its input is the state (4 numbers) and its output layer has one linear neuron per
action (2 Q-values).

## Learning the Q-values: the Bellman target

Nobody gives the network the true Q-values to learn from. The **Bellman equation** relates the Q-value of a state to
the Q-values of the next state: the return from now is the reward now plus the discounted return from the next state.
In an environment with randomness, it holds on average over the possible next states; CartPole is deterministic, so
it holds exactly.

```
Q(state, action) = reward + gamma * max over a' of Q(next_state, a')
Q(state, action) = reward                                  if the episode ended in next_state
```

So every step the agent experienced, `(state, action, reward, next_state)`, gives a target for `Q(state, action)`,
computed with the network's own estimate for the next state. Training the network with the mean squared error towards
these targets makes its estimates consistent with the equation, and the rewards propagate back from step to step.
Only the output for the action that was taken gets a Bellman target. The other output gets its current prediction as
target, so at first its error is zero (`replay_experience` in `dqn.py`).

Only a **terminated** episode has no future: when the episode is **truncated** by the time limit, the pole was still
up and the future would have gone on, so the target still includes `gamma * max Q(next_state)`. That is why the
replay memory stores `terminated`, not `terminated or truncated` (`play_episode`).

## Exploring: epsilon-greedy

If the agent always took the action it currently thinks is best, it would never try the others and could get stuck
with a bad estimate. With the **epsilon-greedy** policy (`choose_action`), it takes a random action with probability
`epsilon` and the best known action otherwise. `epsilon` starts at 1 (only random actions) and is multiplied by
`epsilon_decay` after every episode, down to `epsilon_min`: the agent explores less as it learns.

## Two ideas that make it stable

**Replay memory.** Consecutive steps are almost the same state, so learning from them in order would push the network
towards whatever happens right now and make it forget the rest. The agent stores its steps in a replay memory and
learns from a random sample of it (`replay_experience`), which breaks this correlation and reuses each experience
many times.

**Target network.** The Bellman target depends on the network's own estimate of the next state. If the same network
computed the targets, every update would also move the targets, like chasing your own shadow. A copy of the network,
the **target model**, computes them instead, and it is refreshed with the current weights only every
`target_update_freq` episodes.

**Double DQN** (optional, `double_dqn=True`). Taking the max over noisy estimates tends to pick the ones that are too
high, so DQN overestimates the Q-values. Double DQN splits the job: the model chooses the best next action, and the
target model evaluates it.

## The training loop

`train_dqn` in `dqn.py`, for each episode:

```
1. act        play_episode:       play an episode with epsilon-greedy, storing each
                                   (state, action, reward, next_state, terminated) in the replay memory
2. learn      learn_from_memory:  sample batch_size transitions, compute their Bellman targets
                                   (replay_experience) and fit the model to them
3. evaluate   evaluate_greedy:    every eval_freq episodes, play eval_episodes without exploration;
                                   save the model if it is the best so far
4. schedules                      epsilon = max(epsilon_min, epsilon * epsilon_decay)
5. target                         every target_update_freq episodes, copy the model into the target model
```

### Simplifications in this project

- **When it learns.** This project learns once per episode: it samples 512 transitions and makes one update per
  transition (`batch_size=1` in `model.train`). The original DQN makes one minibatch update after every step of the
  environment.
- **Targets of the actions not taken.** The targets of a batch are computed once, before its 512 updates. Each
  update also changes the predictions for the other transitions, so the targets of the actions not taken (their
  prediction before the updates) become slightly outdated, and the following updates pull those Q-values back towards
  them. The original DQN only trains the Q-value of the action taken.
- **Optimizer and loss.** Plain gradient descent with the mean squared error, with the error clipped to `[-10, 10]`
  (`clip_gradients`; with one transition per update, `dL/dq` is the error itself). The original DQN used RMSProp and
  clipped the error to `[-1, 1]`, which behaves like the Huber loss.

## Evaluation and checkpoints

The reward of the training episodes includes the random actions of the exploration, so it doesn't show what the agent
has learned. Every `eval_freq` episodes, `evaluate_greedy` plays `eval_episodes` with the **greedy** policy (no random
actions) in a separate environment, always from the same starting states, so the evaluations can be compared.

A DQN policy often gets worse after reaching its best (the updates that improve some states can spoil others), so
the last model isn't necessarily the best one. The **checkpoint** keeps the model with the best greedy evaluation,
starting with the model before the training: `train-cartpole.py` saves it to `models/cartpole.joblib`.

## Reading the metrics

Each episode adds a row to `metrics/cartpole.csv`, and `show-cartpole-metrics.py` plots them:

- **total_reward**: the reward of the training episode, with exploration. Noisy.
- **average_rolling_reward**: the mean over the last 100 episodes. Smooth, but it hides drops of the policy.
- **greedy_reward**: the mean reward of the greedy evaluation (only every `eval_freq` episodes). The best measure of
  what the agent has learned.
- **mean_q**: the mean of the highest Q-value of the sampled states. It should level off near the return the agent
  gets, at most `1 / (1 - gamma)` (5 for `gamma = 0.8`). Growing beyond it means the Q-values are overestimated.
- **loss**: the error towards the Bellman targets. The targets move with the model, so a falling loss doesn't mean the
  agent is learning to balance; look at the rewards.
- **epsilon**, **learning_rate**: the schedules. **steps** equals **total_reward** in CartPole (a reward of 1 per step).

## What to expect

These are the results of `train-cartpole.py` over 9 seeds:

- The greedy reward grows from about 90 after 100 episodes to about 420 after 500 episodes. None of the runs diverged.
- The mean Q-value levels off just above 5.0 (about 5.02 to 5.05). The true Q-values are always below
  `1 / (1 - gamma) = 5`, so this is a mild overestimation, the effect Double DQN reduces.
- The policy doesn't settle: over 1000 episodes, 3 of the 9 runs fell below 300 after reaching 500. Keeping the best
  checkpoint makes the saved model better than the last one (457 against 422 on average, on episodes it never saw).
- Within 500 episodes, `gamma = 0.8` learned the best. Higher values look further ahead, but their Q-values take
  longer to grow and the reward stayed lower (about 320 with 0.95 and 210 to 250 with 0.99). Double DQN didn't help
  at `gamma = 0.8`.

## Things to try

- `gamma = 0.99` with more episodes, with and without `double_dqn=True`, and compare `mean_q` with `1 / (1 - gamma)`.
- A smaller `target_update_freq`, or a target model updated after every episode.
- Learning after every step instead of once per episode, as in the original DQN.
- The Adam optimizer for the Q-network (`model.configure(loss="mse", optimizer="adam")`) with a smaller learning rate.
