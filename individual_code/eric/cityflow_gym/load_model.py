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
num_episodes = 100
env = DummyVecEnv([lambda: LineEnv('poundsign', steps_per_episode)])

model = PPO2.load("ppo2_line")
obs = env.reset()
env.render()
total_rewards = 0
for i in range(steps_per_episode):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  total_rewards+= rewards[0]
  if done:
    break
print(total_rewards)