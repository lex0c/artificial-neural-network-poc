import gymnasium as gym

from cli import keep_dashboard_open, start_dashboard, display_path, load_trained_model, project_path, training_args
from data import set_seed
from dqn import train_dqn
from feedforward import clone_model


# Continues training the model saved by train-cartpole.py, exploring less (epsilon fixed at 0.1) and with a small
# L1/L2 regularization. Saves the best model to models/cartpole-checkpoint.joblib (starting with the loaded one, so it
# is never worse than it) and the last one to models/cartpole-finetuned.joblib.

args = training_args("Continues training the DQN model of train-cartpole.py.", render=True, episodes=500)

env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
set_seed(args.seed, env)
eval_env = gym.make("CartPole-v1")  # a separate environment for the greedy evaluations


model = load_trained_model(project_path('models', 'cartpole.joblib'), 'train-cartpole.py')
model.summary()
monitor = start_dashboard(args.dashboard, {"Q-network": model})

target_model = clone_model(model)


train_dqn(
    env, model, target_model,
    episodes=args.episodes,
    batch_size=128,
    gamma=0.80,
    learning_rate=0.001,
    learning_rate_decay=0.99,
    learning_rate_decay_freq=300,
    epsilon=0.1,
    epsilon_min=0.1,
    l1_lambda=0.0001,
    l2_lambda=0.0001,
    target_update_freq=10,
    double_dqn=False,
    eval_env=eval_env,
    eval_freq=50,
    eval_episodes=10,
    metrics_filepath=project_path('metrics', 'cartpole.csv'),  # plotted by show-metrics.py
    checkpoint_filepath=project_path('models', 'cartpole-checkpoint.joblib'),
)


finetuned_filepath = project_path('models', 'cartpole-finetuned.joblib')
model.save(finetuned_filepath)
print(f"Last model saved to {display_path(finetuned_filepath)}")
env.close()
eval_env.close()
keep_dashboard_open(monitor)
