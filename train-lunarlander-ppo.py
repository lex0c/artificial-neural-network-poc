import gymnasium as gym

from cli import keep_dashboard_open, start_dashboard, project_path, training_args
from data import set_seed
from feedforward import FeedForward
from ppo import train_ppo


# Trains a PPO agent on LunarLander (see PPO.md) and saves the best policy to models/lunarlander-ppo.joblib.
#
# A lander has to touch down gently between two flags, firing its main engine or one of its two side engines. The
# reward guides it: points for getting closer to the pad and slowing down, +10 for each leg on the ground, +100 for
# landing and -100 for crashing, and a small cost for each firing of an engine. 200 points count as solved.
#
# Needs Box2D (see the README).

args = training_args("Trains a PPO agent on LunarLander and saves the best policy to models/lunarlander-ppo.joblib.",
                     watch=True, steps=1_000_000)

env = gym.make("LunarLander-v3")
set_seed(args.seed, env)
eval_env = gym.make("LunarLander-v3")  # a separate environment for the evaluations


def network(num_outputs, output_act_fn, loss):
    model = FeedForward()
    model.add_layer(num_inputs=env.observation_space.shape[0], num_neurons=64, act_fn='tanh')
    model.add_layer(num_inputs=64, num_neurons=64, act_fn='tanh')
    model.add_layer(num_inputs=64, num_neurons=num_outputs, act_fn=output_act_fn)
    model.configure(loss=loss, optimizer='adam')
    return model  # no normalization: the values of the state are already around -1 to 1


# The state (position, velocity, angle, angular velocity, and whether each leg touches the ground) in; the policy gives
# one probability per action (nothing, left engine, main engine, right engine), and the value model the return
# expected from the state
policy = network(env.action_space.n, 'softmax', None)  # its gradient comes from the PPO objective
value_model = network(1, 'linear', 'mse')
policy.summary()

if args.watch:
    from viewer import LiveViewer
    LiveViewer("LunarLander-v3", policy)  # before the dashboard (see viewer.py)

monitor = start_dashboard(args.dashboard, {"policy": policy, "value model": value_model})


train_ppo(
    env, policy, value_model,
    total_steps=args.steps,
    learning_rate=0.0003,
    num_steps=2048,          # steps played with the current policy in each iteration...
    num_epochs=10,           # ...then 10 passes over them...
    minibatch_size=64,       # ...in minibatches of 64
    gamma=0.999,             # the +100 of a landing comes hundreds of steps later, so the agent has to look far ahead
    gae_lambda=0.98,
    clip_range=0.2,          # each update can change the probability of an action by about 20% at most
    entropy_coef=0.01,       # a bonus for keeping some randomness, so the policy doesn't stop exploring too early
    eval_env=eval_env,
    eval_freq=10,            # iterations between evaluations
    eval_episodes=10,
    metrics_filepath=project_path('metrics', 'lunarlander-ppo.csv'),  # plotted by show-metrics.py
    checkpoint_filepath=project_path('models', 'lunarlander-ppo.joblib'),  # keeps the policy with the best evaluation
)


env.close()
eval_env.close()
keep_dashboard_open(monitor)
