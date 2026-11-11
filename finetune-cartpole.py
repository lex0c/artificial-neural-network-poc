import gymnasium as gym

from data import set_seed
from dqn import train_dqn
from feedforward import load_model, clone_model


# Continues training the model saved by train-cartpole.py, exploring less (epsilon fixed at 0.1) and with a small
# L1/L2 regularization. Saves the best model to models/cartpole-checkpoint.joblib (starting with the loaded one, so it
# is never worse than it) and the last one to models/cartpole-finetuned.joblib.

SEED = None     # set an integer (e.g. 0) to make the run reproducible
RENDER = False  # True to watch the training episodes (much slower: rendering runs at about 50 steps per second)

env = gym.make("CartPole-v1", render_mode="human" if RENDER else None)
set_seed(SEED, env)
eval_env = gym.make("CartPole-v1")  # a separate environment for the greedy evaluations


model = load_model('models/cartpole.joblib')
model.summary()

target_model = clone_model(model)


train_dqn(
    env, model, target_model,
    episodes=500,
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
    metrics_filepath='metrics/cartpole.csv',  # plotted by show-metrics.py
    checkpoint_filepath='models/cartpole-checkpoint.joblib',
)


model.save('models/cartpole-finetuned.joblib')
env.close()
eval_env.close()
