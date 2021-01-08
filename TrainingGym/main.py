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
import os

def main():
    with open('global_config.json') as global_json_file:
        local_config_path = json.load(global_json_file)['config_path']
    with open(f'Scenarios/{local_config_path}') as json_file:
        config_params = json.load(json_file)

    num_proc = config_params['num_proc']
    steps_per_episode = config_params['steps_per_episode']
    num_episodes = config_params['num_episodes']
    controlled_lights = config_params['controlled_lights']

    # Load existing light models if they exist
    # models = []
    # for light in controlled_lights:
    #     path_name = f'Scenarios/{config_params["model_save_path"]}/PPO2_{light["name"]}'
    #     if os.path.isfile(path_name):
    #         models.append(PPO2.load(path_name))
    #     else:
    #         models.append(None)
    # print(models)



    # for i in range(len(models)):
    #     if models[i] is None:
    #         models[i] = PPO2(MlpPolicy, env, verbose=1)
    # model = PPO2(MlpPolicy, env, verbose=1)

    for i in range(len(controlled_lights)):
        env = create_env(controlled_lights[i]['name'], num_proc, steps_per_episode)
        model = PPO2(MlpPolicy, env, verbose=1)
        light = controlled_lights[i]
        start = time.time()
        model.learn(total_timesteps=steps_per_episode * num_episodes)
        print(f'LEARNING TIME: {time.time() - start}')
        model.save(f'Scenarios/{config_params["model_save_path"]}/PPO2_{light["name"]}')
        print('done learning')
        env.close()
        del env

def create_env(node_name, num_proc, steps_per_episode):
    if num_proc == 1:
        env = DummyVecEnv([lambda: SumoEnvParallel(steps_per_episode, False, node_name)])
    else:
        if sys.platform == 'win32':
            thread_method = 'spawn'
        else:
            thread_method = 'forkserver'  # Fork was having issues with multi-agent for me so I switched to forkserver
        env = SubprocVecEnv([lambda: SumoEnvParallel(steps_per_episode, False, node_name) for i in range(num_proc)],
                            start_method=thread_method)
    return env

if __name__ == '__main__':
    main()
