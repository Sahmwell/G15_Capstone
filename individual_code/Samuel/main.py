# Remove TF warnings in Stable baselines (may not be safe)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from env.SumoEnvParallel import SumoEnvParallel
import time
import sys
import json

if __name__ == '__main__':
    with open('global_config.json') as global_json_file:
        local_config_path = json.load(global_json_file)['config_path']
    with open(local_config_path) as json_file:
        config_params = json.load(json_file)

    num_proc = config_params['num_proc']
    steps_per_episode = config_params['steps_per_episode']
    num_episodes = config_params['num_episodes']
    if num_proc == 1:
        env = DummyVecEnv([lambda: SumoEnvParallel(steps_per_episode, False)])
    else:
        if sys.platform == 'win32':
            thread_method = 'spawn'
        else:
            thread_method = 'fork'
        env = SubprocVecEnv([lambda: SumoEnvParallel(steps_per_episode, False) for i in range(num_proc)],
                            start_method=thread_method)

    model = PPO2(MlpPolicy, env, verbose=1)
    start = time.time()
    model.learn(total_timesteps=steps_per_episode * num_episodes)
    print(f'LEARNING TIME: {time.time() - start}')
    model.save(config_params['model_save_path'])
    print('done learning')
