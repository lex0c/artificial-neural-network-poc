import gymnasium as gym

from feedforward import FeedForward, clone_model
from dqn import train_dqn
from etc import set_seed


SEED = None  # set an integer (e.g. 0) to make the run reproducible

env = gym.make("CartPole-v1", render_mode="human")
set_seed(SEED, env)


model = FeedForward()
model.add_layer(num_inputs=env.observation_space.shape[0], num_neurons=64, act_fn='relu')
model.add_layer(num_inputs=64, num_neurons=env.action_space.n, act_fn='linear')
model.configure(loss="mse")
model.summary()

target_model = clone_model(model)
target_model.summary()


train_dqn(
    env, model, target_model,
    episodes=500,
    batch_size=512,
    gamma=0.80, # higher values are stable, but learn slower within 500 episodes
    learning_rate=0.001,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.990,
    target_update_freq=10,
    double_dqn=False, # didn't help at this gamma
)


model.save('models/cartpole.joblib')
env.close()
