import math
import time
from datetime import datetime

import numpy as np

from progress import display_path, format_duration, timing
from rl import evaluate_greedy, write_to_csv


# Proximal Policy Optimization (PPO, Schulman et al. 2017): an agent learns a policy, the probability of each action in
# each state, directly. Two FeedForward models work together:
# - the policy: the state in, one probability per action out (softmax). The agent samples its actions from it.
# - the value model (the critic): the state in, V(state) out, the return expected from that state with the current
#   policy. It only serves to judge whether an action turned out better or worse than expected.
# PPO.md explains the concepts and how this training loop works.


# Plays num_steps steps with the current policy, sampling each action from its probabilities, and returns them as
# arrays. Episodes continue from one call to the next: state and episode_reward are where the previous call stopped
# (None and 0 to start a new episode).
#
# next_states holds the state returned by each step, also when the episode ended there (before the reset), because a
# truncated episode still needs the value of its last state.
#
# Returns the steps, the state and reward of the episode in progress, and the total rewards of the finished episodes.
def collect_rollout(env, policy, num_steps, state=None, episode_reward=0.0):
    if state is None:
        state, _ = env.reset()

    rollout = {key: [] for key in ("states", "actions", "action_probs", "rewards", "next_states", "terminated", "truncated")}
    finished_rewards = []
    total_reward = episode_reward

    for _ in range(num_steps):
        probs = policy.predict(state.reshape(1, -1))[0]
        action = np.random.choice(len(probs), p=probs)

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        for key, value in zip(rollout, (state, action, probs[action], reward, next_state, terminated, truncated)):
            rollout[key].append(value)

        if terminated or truncated:
            finished_rewards.append(total_reward)
            total_reward = 0.0
            next_state, _ = env.reset()

        state = next_state

    return {key: np.array(values) for key, values in rollout.items()}, state, total_reward, finished_rewards


# Generalized Advantage Estimation (GAE, Schulman et al. 2016): how much better each action turned out than the value
# model expected. The one-step estimate is the TD error
#   delta[t] = reward[t] + gamma * V(next_state[t]) - V(state[t])     (no future value if terminated)
# and the advantage sums the following ones, each discounted by gamma * lambda:
#   advantage[t] = delta[t] + gamma * lambda * advantage[t + 1]       (until the episode ends)
# lambda balances the two extremes: 0 uses only the one-step estimate (low variance, but it trusts V), and 1 the real
# rewards until the end of the episode (no trust in V, but noisy). The returns, advantage + V, are the targets of V.
def compute_advantages(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    advantages = np.zeros(len(rewards))
    next_advantage = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]

        # After the last step of an episode (terminated or truncated), the next step belongs to another episode
        episode_continues = 1 - (terminated[t] or truncated[t])
        next_advantage = delta + gamma * gae_lambda * episode_continues * next_advantage
        advantages[t] = next_advantage

    return advantages, advantages + values


# Gradient of the PPO loss with respect to z of the softmax output layer of the policy, for a minibatch: the delta that
# policy.backward() needs. PPO maximizes, over the samples,
#   min(ratio * advantage, clip(ratio, 1 - clip_range, 1 + clip_range) * advantage) + entropy_coef * entropy
# where ratio = p(action) / p_old(action) compares the policy with the one that collected the data. The loss is minus
# this mean. The clip removes the incentive to move the ratio beyond 1 +- clip_range: a policy that changes too much
# from data collected by another one would no longer be estimated well.
#
# The chain rule, with p = softmax(z):
#   d ratio / d z[j]  = ratio * (one_hot[j] - p[j])        (because d log p(action) / d z[j] = one_hot[j] - p[j])
#   d entropy / d z[j] = -p[j] * (log p[j] + entropy)      (entropy = -sum of p * log p)
# The min only lets the gradient through the unclipped term: when clipping makes the term smaller, its gradient is 0.
#
# Returns the delta and some statistics: the loss, the mean entropy, and the fraction of samples whose term was clipped.
def ppo_output_delta(probs, actions, advantages, old_action_probs, clip_range, entropy_coef):
    rows = np.arange(len(actions))
    ratio = probs[rows, actions] / old_action_probs

    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1 - clip_range, 1 + clip_range) * advantages
    through = unclipped <= clipped  # the samples whose min is the unclipped term

    one_hot = np.zeros_like(probs)
    one_hot[rows, actions] = 1
    grad_objective = (through * advantages * ratio)[:, None] * (one_hot - probs)

    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    entropy = -np.sum(probs * log_probs, axis=1)
    grad_entropy = -probs * (log_probs + entropy[:, None])

    delta = -(grad_objective + entropy_coef * grad_entropy) / len(actions)  # minus, because the loss is minimized
    loss = -np.mean(np.minimum(unclipped, clipped) + entropy_coef * entropy)

    return delta, {"loss": loss, "entropy": float(np.mean(entropy)), "clip_fraction": float(np.mean(~through))}


# Trains a PPO agent for total_steps steps of the environment, updating the policy and the value model in place. Each
# iteration:
#   1. collect: play num_steps steps with the current policy (the last iteration plays the steps left, when num_steps
#      doesn't divide total_steps)
#   2. advantages: compute them with GAE, using the value model
#   3. learn: num_epochs passes over the collected steps, in random minibatches of at most minibatch_size steps; each
#      minibatch updates the policy with the PPO gradient and the value model towards the returns (mean squared error)
#   4. evaluate: every eval_freq iterations, play eval_episodes with the most likely action in eval_env, and save the
#      checkpoint when it is the best so far (starting with the policy before the training). A training shorter than
#      eval_freq iterations is evaluated at its end, so its checkpoint isn't left with the untrained policy.
# The data is then thrown away: PPO only learns from steps played with the current policy (on-policy).
#
# Both models need configure(loss=None, optimizer=...) for the policy (its gradient comes from ppo_output_delta) and
# configure(loss="mse", ...) for the value model. Returns the total reward of each finished episode.
def train_ppo(env, policy, value_model, total_steps, learning_rate=0.0003, num_steps=2048, minibatch_size=64,
              num_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, entropy_coef=0.0,
              eval_env=None, eval_freq=5, eval_episodes=10, checkpoint_filepath=None, metrics_filepath=None, log_every=5):
    if checkpoint_filepath and eval_env is None:
        raise ValueError("checkpoint_filepath needs eval_env: the checkpoint is the policy with the best evaluation")

    for name, value in (("total_steps", total_steps), ("num_steps", num_steps), ("minibatch_size", minibatch_size)):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")

    episode_rewards = []
    metrics_key = datetime.now().strftime('%Y%m%d%H%M%S')
    state, episode_reward = None, 0.0
    training_start = time.time()
    block_rewards = []  # rewards of the episodes finished since the last progress line

    if eval_env is not None:
        best_greedy_reward, best_iteration, best_steps = evaluate_greedy(eval_env, policy, eval_episodes), 0, 0
        print(f"Greedy evaluation before training: {best_greedy_reward:.1f}")

        if checkpoint_filepath:
            policy.save(checkpoint_filepath)

    num_iterations = math.ceil(total_steps / num_steps)  # rounded up: the last iteration plays the steps left
    steps_done = 0

    for iteration in range(1, num_iterations + 1):
        start_time = time.time()
        rollout_steps = min(num_steps, total_steps - steps_done)
        steps_done += rollout_steps

        # 1. Collect
        rollout, state, episode_reward, finished = collect_rollout(env, policy, rollout_steps, state, episode_reward)
        episode_rewards += finished
        block_rewards += finished

        # 2. Advantages
        values = value_model.predict(rollout["states"])[:, 0]
        next_values = value_model.predict(rollout["next_states"])[:, 0]
        advantages, returns = compute_advantages(rollout["rewards"], values, next_values, rollout["terminated"], rollout["truncated"], gamma, gae_lambda)

        # 3. Learn
        policy_states = policy.normalize(rollout["states"])
        value_states = value_model.normalize(rollout["states"])
        stats = []

        for epoch in range(num_epochs):
            # The steps in a random order, cut into minibatches of minibatch_size (the last one smaller when
            # minibatch_size doesn't divide the steps, and a single one when the rollout is shorter than a minibatch)
            order = np.random.permutation(rollout_steps)

            for start in range(0, rollout_steps, minibatch_size):
                batch = order[start:start + minibatch_size]
                # Advantages normalized per minibatch: only their relative size matters, and it keeps the step stable
                batch_advantages = advantages[batch]
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                probs = policy.forward_layers(policy_states[batch])
                delta, batch_stats = ppo_output_delta(probs, rollout["actions"][batch], batch_advantages, rollout["action_probs"][batch], clip_range, entropy_coef)
                policy.update(policy.backward(delta), learning_rate)

                batch_stats["value_loss"] = value_model.train_step(value_states[batch], returns[batch, None], learning_rate)
                stats.append(batch_stats)

        # 4. Evaluate
        greedy_reward = None

        if eval_env is not None and (iteration % eval_freq == 0 or (iteration == num_iterations and num_iterations < eval_freq)):
            greedy_reward = evaluate_greedy(eval_env, policy, eval_episodes)

            if greedy_reward > best_greedy_reward:
                best_greedy_reward, best_iteration, best_steps = greedy_reward, iteration, steps_done

                if checkpoint_filepath:
                    policy.save(checkpoint_filepath)

        mean_reward = float(np.mean(finished)) if finished else None
        metrics = {key: float(np.mean([s[key] for s in stats])) for key in ("loss", "value_loss", "entropy", "clip_fraction")}

        policy.notify("on_metrics", {'iteration': iteration, 'mean_episode_reward': mean_reward, 'greedy_reward': greedy_reward,
                                     'value_loss': metrics["value_loss"], 'entropy': metrics["entropy"],
                                     'clip_fraction': metrics["clip_fraction"]})

        if metrics_filepath:
            write_to_csv(metrics_filepath, {
                'metrics_key': metrics_key,
                'iteration': iteration,
                'steps': steps_done,
                'episodes': len(episode_rewards),
                'mean_episode_reward': mean_reward,
                'greedy_reward': greedy_reward,
                'policy_loss': metrics["loss"],
                'value_loss': metrics["value_loss"],
                'entropy': metrics["entropy"],
                'clip_fraction': metrics["clip_fraction"],
                'duration': time.time() - start_time,
            })

        # Progress: one line every log_every iterations, with the episodes finished in them
        if iteration % log_every == 0 or iteration == num_iterations:
            reward_text = f"{np.mean(block_rewards):.1f} ({len(block_rewards)} episodes)" if block_rewards else "- (no episode ended)"
            print(f"Iteration {iteration}/{num_iterations} ({steps_done:,} steps) - episode reward {reward_text} "
                  f"- entropy {metrics['entropy']:.3f} - clip fraction {metrics['clip_fraction']:.3f} - {timing(training_start, iteration, num_iterations)}")
            block_rewards = []

        if greedy_reward is not None:
            print(f"  Greedy evaluation: {greedy_reward:.1f} (best: {best_greedy_reward:.1f})")

    print(f"\nDone in {format_duration(time.time() - training_start)}.")

    if eval_env is not None:
        when = f"after iteration {best_iteration} ({best_steps:,} steps)" if best_iteration else "before training"
        print(f"Best greedy evaluation: {best_greedy_reward:.1f} ({when})")

    if checkpoint_filepath:
        print(f"Best policy saved to {display_path(checkpoint_filepath)}")

    if metrics_filepath:
        print(f"Metrics saved to {display_path(metrics_filepath)} (plot them with: python show-metrics.py {display_path(metrics_filepath)})")

    return episode_rewards
