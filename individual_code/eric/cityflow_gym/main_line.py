# Remove TF warnings in Stable baselines (may not be safe)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.LineEnv import LineEnv


steps_per_episode = 1000
num_episodes = 10

env = DummyVecEnv([lambda: LineEnv('line', steps_per_episode)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=steps_per_episode*num_episodes)
model.save('ppo2_line')
print('done learning')

obs = env.reset()
env.render()
for i in range(steps_per_episode):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  if done:
    break
  print(rewards)