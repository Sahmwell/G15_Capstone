# Remove TF warnings in Stable baselines (may not be safe)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv , SubprocVecEnv
from stable_baselines import PPO2

from env.SumoEnv import SumoEnv

import time

steps_per_episode = 1000
num_episodes = 20
env = DummyVecEnv([lambda: SumoEnv(steps_per_episode, False)])
# env = SubprocVecEnv([SumoEnv(steps_per_episode, False) for i in range(12)], start_method='spawn')

model = PPO2(MlpPolicy, env, verbose=1)
start = time.time()
model.learn(total_timesteps=steps_per_episode*num_episodes)
print(f'LEARNING TIME: {time.time() - start}')
model.save('ppo2_pound')
print('done learning')

env.render()
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