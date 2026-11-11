import gymnasium as gym

from feedforward import load_model, clone_model
from dqn import train_dqn
from etc import set_seed


SEED = None  # set an integer (e.g. 0) to make the run reproducible

env = gym.make("CartPole-v1", render_mode="human")
set_seed(SEED, env)


model = load_model('models/cartpole.joblib')
model.summary()

target_model = clone_model(model)
target_model.summary()


train_dqn(
    env, model, target_model,
    episodes=500,
    batch_size=128,
    gamma=0.80, # higher values are stable, but learn slower within 500 episodes
    learning_rate=0.001,
    learning_rate_decay=0.99,
    learning_rate_decay_freq=300,
    epsilon=0.1,
    epsilon_min=0.1,
    l1_lambda=0.0001,
    l2_lambda=0.0001,
    target_update_freq=10,
    double_dqn=False, # didn't help at this gamma
    checkpoint_filepath='models/cartpole-checkpoint.joblib',
)


model.save('models/cartpole-finetuned.joblib')
env.close()
