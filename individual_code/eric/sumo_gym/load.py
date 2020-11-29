# Remove TF warnings in Stable baselines (may not be safe)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.SumoEnv import SumoEnv


steps_per_episode = 1000
num_episodes = 100
env = DummyVecEnv([lambda: SumoEnv(steps_per_episode, True, 0)])

model = PPO2.load("ppo2_pound")
obs = env.reset()
total_rewards = 0
for i in range(steps_per_episode):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  total_rewards+= rewards[0]
  if done:
    break
print(total_rewards)