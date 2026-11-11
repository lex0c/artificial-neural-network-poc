# Proximal Policy Optimization (PPO)

`ppo.py` trains an agent with PPO (Schulman et al. 2017). Where [DQN](DQN.md) learns the value of each action and then
takes the best one, PPO learns the **policy** directly: the probability of each action in each state. This page
explains the ideas and how the training loop works, using the same CartPole and Acrobot environments as DQN.md, and
LunarLander.

## Two networks: the policy and the critic

- The **policy** is a `FeedForward` model with a softmax output: the state in, one probability per action out. The
  agent samples its actions from these probabilities, so exploration is built in: an action the policy isn't sure
  about still gets tried, and the policy itself learns how random to be in each state.
- The **value model**, or critic, is a `FeedForward` model with one linear output: `V(state)`, the return expected from
  that state with the current policy. It doesn't choose anything; it serves to judge whether an action turned out
  better or worse than expected.

Both have two hidden layers of 64 `tanh` neurons in `train-cartpole-ppo.py` and `train-acrobot-ppo.py`.

## Advantages: better or worse than expected

The **advantage** of an action is how much more return it brought than the value model expected from that state. A
positive advantage means the action should become more likely, and a negative one less likely.

The simplest estimate looks one step ahead (the TD error):

```
delta[t] = reward[t] + gamma * V(next_state[t]) - V(state[t])      (no V(next_state) if the episode terminated)
```

It trusts `V`, which may be wrong. Summing the real rewards until the end of the episode doesn't trust `V`, but it is
noisy. **Generalized Advantage Estimation** (GAE, `compute_advantages`) blends the two with a factor `lambda`:

```
advantage[t] = delta[t] + gamma * lambda * advantage[t + 1]          (until the episode ends)
```

With `lambda = 0` it is the one-step estimate, and with `lambda = 1` the real rewards. The returns, `advantage + V`,
are the targets the value model learns.

For example, with 3 steps of reward 1, every value `0.5`, `gamma = 0.5`, `lambda = 0.5`, and an episode that
terminates at the last step:

```
delta     = [1 + 0.5 * 0.5 - 0.5,  1 + 0.5 * 0.5 - 0.5,  1 - 0.5] = [0.75, 0.75, 0.5]
advantage = [0.75 + 0.25 * 0.875,  0.75 + 0.25 * 0.5,    0.5    ] = [0.96875, 0.875, 0.5]
```

`test_ppo.py` checks these numbers, and also a truncated episode: as in DQN, an episode cut by the time limit didn't
end, so its last step still adds `gamma * V` of the state where it stopped.

## The policy update

The basic policy gradient increases the log-probability of each action in proportion to its advantage. PPO adds a
safeguard. It compares the current policy with the one that collected the data through the ratio

```
ratio = p(action) / p_old(action)
```

and maximizes

```
min(ratio * advantage, clip(ratio, 1 - clip_range, 1 + clip_range) * advantage) + entropy_coef * entropy
```

The data was collected by the old policy, so it only says something reliable about policies close to it. The clip
removes the incentive to push the ratio beyond `1 +- clip_range` (`0.2`): once an action is 20% more likely than
before, a positive advantage stops pushing it further. The **entropy** bonus (`-sum of p * log p`) rewards keeping
some randomness, so the policy doesn't become deterministic before it has explored enough.

This objective isn't one of the losses in `losses.py`, so `ppo_output_delta` computes its gradient with respect to
`z` of the softmax layer by hand, with the chain rule:

```
d ratio / d z[j]   = ratio * (one_hot[j] - p[j])           because d log p(action) / d z[j] = one_hot[j] - p[j]
d entropy / d z[j] = -p[j] * (log p[j] + entropy)
```

The min only lets the gradient through the unclipped term; when clipping makes the term smaller, its gradient is 0.
This delta goes to `policy.backward()`, the same backpropagation the other examples use, and `policy.update()` applies
it with Adam. `test_ppo.py` checks it against the gradient measured numerically.

## The training loop

`train_ppo` in `ppo.py`, for each iteration:

```
1. collect      collect_rollout:      play num_steps steps with the current policy, sampling the actions
2. advantages   compute_advantages:   GAE with the value model (V of every state and next state)
3. learn                              num_epochs passes over the steps, in random minibatches:
                ppo_output_delta        the policy takes a step of the PPO gradient
                value_model.train_step  the value model takes a step towards the returns (mean squared error)
4. evaluate     evaluate_greedy:      every eval_freq iterations, play eval_episodes with the most likely action;
                                      save the policy if it is the best so far
```

The steps are then thrown away: PPO only learns from data played with the current policy (**on-policy**), unlike
DQN, which keeps a replay memory and reuses old steps many times. The advantages are normalized in each minibatch
(mean 0, standard deviation 1): only their relative size matters, and it keeps the size of the steps stable.

### Simplifications in this project

- One environment at a time. Common implementations play several copies of the environment in parallel.
- None of the common extras: no clipping of the gradient norm, no clipping of the value updates, no decreasing
  learning rate and no early stop when the policy changes too much.
- A decreasing learning rate was measured on LunarLander, over 9 seeds: falling linearly to 0, it halved the
  oscillation of the training reward at the end, but the policies scored the same, about 270 on 100 new episodes with
  either schedule, because the checkpoint already keeps the best policy. In one run, the smaller steps also made the
  policy take about 200 iterations to recover from a relapse. A steadier curve isn't a better agent.

## Hyperparameters

| Parameter | Value | Meaning |
| --- | --- | --- |
| `num_steps` | 2048 | steps collected per iteration |
| `num_epochs` | 10 | passes over them |
| `minibatch_size` | 64 | steps per gradient step |
| `learning_rate` | 0.0003 | Adam, for both networks |
| `gamma` | 0.99 | discount factor |
| `gae_lambda` | 0.95 | GAE: between the one-step estimate (0) and the real rewards (1) |
| `clip_range` | 0.2 | how far the ratio can usefully move in one iteration |
| `entropy_coef` | 0.01 | bonus for keeping some randomness |

These are the usual defaults of PPO, and the same values work for CartPole and Acrobot, which is one of the strengths
of PPO: DQN needed different hyperparameters for each one. LunarLander changes only the horizon, with `gamma=0.999` and
`gae_lambda=0.98`: the +100 of a landing comes hundreds of steps after the decisions that lead to it.

## Reading the metrics

Each iteration adds a row to the metrics file, and `python show-metrics.py metrics/acrobot-ppo.csv` plots them:

- **mean_episode_reward**: the mean reward of the episodes that ended in the iteration, with the sampled actions.
- **greedy_reward**: the evaluation with the most likely action (every `eval_freq` iterations).
- **entropy**: it starts at `log(number of actions)` (a uniform policy: 1.10 for the 3 actions of Acrobot) and falls
  as the policy becomes confident.
- **clip_fraction**: the fraction of samples whose term was clipped. A few percent is normal; much more means the
  policy changes too much in each iteration.
- **value_loss**: the error of the value model. It peaks when the rewards change quickly, because the values it
  learned no longer hold.
- **policy_loss**: minus the clipped objective. Its value isn't a measure of progress (the advantages are normalized).

## What to expect

Over 9 seeds:

- **CartPole** (`train-cartpole-ppo.py`, 100,000 steps, about 40 seconds): every run reaches the maximum of 500, first
  between 10,000 and 41,000 steps, and stays there.
- **Acrobot** (`train-acrobot-ppo.py`, 300,000 steps, about 2 minutes): every run learns. The saved policies average
  -85.5 (all between -78 and -105) and the last ones -86, close to the -100 usually considered solved, and they stay
  stable after learning. The training reward passed -150 between 50,000 and 180,000 steps.
- **LunarLander** (`train-lunarlander-ppo.py`, 1,000,000 steps, about 4 minutes): every run solves it. The greedy
  evaluation passed 200 between 120,000 and 310,000 steps. On 100 episodes the training never evaluated, the saved
  policies average 272 (between 254 and 283) and score 200 or more in 99% of them; the last policies average 270, with
  97%. A few episodes still end in a crash.
- **Without the entropy bonus** (`entropy_coef=0`), 7 of the 9 Acrobot runs learned: one never left -500, and one was
  still improving at the end. With nothing found yet, the advantages are only noise, and the policy can become
  deterministic before it finds the goal.

## DQN or PPO?

### In this project

Measured over 9 seeds each, with the greedy policy on episodes the agents never saw:

| | DQN (`dqn.py`) | PPO (`ppo.py`) |
| --- | --- | --- |
| CartPole | about 415 after 500 episodes (88,000 to 161,000 steps) and 450 after 1,000, oscillating | 500 in every run, first reached within 10,000 to 41,000 steps, and it stays there |
| Acrobot | 2 to 7 good runs of 9 with the schedule of this project. Learning after every step: the best checkpoint between -84 and -104 in 8 runs within 50,000 steps, but the last policy back at -500 in 6 | every run learns (about -85, passing -150 within 50,000 to 180,000 steps) and stays stable |
| Hyperparameters | tuned for each environment | the same for both |

PPO was more reliable, more stable and easier to tune, although DQN got much more tuning effort. When DQN did learn
Acrobot, it needed fewer steps of the environment than PPO: that is the reuse of old data that the general comparison
below refers to. On CartPole, the DQN of this project needed more steps than PPO.

Two limits to this verdict:

- The DQN here is simplified (see "Simplifications in this project" in DQN.md). Modern variants, such as Rainbow
  (Hessel et al. 2018), which combines Double DQN, n-step returns, prioritized replay and other improvements, are much
  more stable.
- Two simple environments and 9 seeds aren't a general comparison of the two methods.

### In general

| | DQN | PPO |
| --- | --- | --- |
| Learns | the value of each action (Q) | the probability of each action, and a critic V |
| Actions | discrete only | discrete or continuous (a continuous action needs another output, e.g. a mean and a spread) |
| Data | replay memory, each step reused many times | collected with the current policy, used for a few epochs, then thrown away |
| Steps of the environment needed | usually fewer: each step is reused | usually more |
| Exploration | from outside (epsilon-greedy) | built into the policy |
| Stability | can oscillate or diverge (target model, many tricks) | stable: each update can only move the policy a little |
| Hyperparameters | sensitive, tuned per environment | the defaults usually work |

PPO is usually the safer default when playing the environment is cheap, as in these simulations. DQN's reuse of old
steps pays off when each step of the environment is expensive.

## Things to try

- `entropy_coef=0` on Acrobot, and compare the entropy curves of the runs that learn and the ones that don't.
- `gae_lambda=0` (only the one-step estimate) and `gae_lambda=1` (only the real rewards).
- `gamma=0.99` on LunarLander: a shorter horizon than the landing needs.
- A larger `clip_range` (e.g. 0.5) and watch `clip_fraction` and the stability of the rewards.
- MountainCar, whose goal a random policy practically never reaches: PPO's exploration isn't enough there either.
