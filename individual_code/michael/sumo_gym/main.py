# Remove TF warnings in Stable baselines (may not be safe)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv , SubprocVecEnv
from stable_baselines import PPO2

from env.SumoEnv import SumoEnv
from env.SumoEnv_Parallel import SumoEnv_Parallel

import time


if __name__ == '__main__':
    # Specify number of processes as first command line argument
    if len(sys.argv) > 1:
        num_proc = int(sys.argv[1])
        assert(num_proc <= 12) # I have not programmed it to support any more than this
    else:
        num_proc = 1

    steps_per_episode = 1000
    num_episodes = 20
    if(num_proc == 1):
        env = DummyVecEnv([lambda: SumoEnv(steps_per_episode, False)])
    else:
        env = SubprocVecEnv([lambda: SumoEnv_Parallel(steps_per_episode, False, i) for i in range(num_proc)], start_method='forkserver')

    model = PPO2(MlpPolicy, env, verbose=1)
    start = time.time()
    model.learn(total_timesteps=steps_per_episode*num_episodes)
    print(f'LEARNING TIME: {time.time() - start}')
    model.save('ppo2_pound')
    print('done learning')