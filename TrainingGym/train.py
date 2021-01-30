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
    # Load configs
    with open('global_config.json') as global_json_file:
        global_config_params = json.load(global_json_file)
        local_config_path = global_config_params['config_path']
    with open(f'Scenarios/{local_config_path}') as json_file:
        config_params = json.load(json_file)

    # Load Config Parameters
    num_proc = global_config_params['num_proc']
    steps_per_episode = config_params['steps_per_episode']
    num_episodes = config_params['num_episodes']
    controlled_lights = config_params['controlled_lights']
    for i in range(len(controlled_lights) - 1, -1, -1):
        if not controlled_lights[i]['train']:
            del controlled_lights[i]
    num_trials = config_params['num_trials']

    # Run a learning session on each light

    for i in range(num_trials * len(controlled_lights)):
        learning_light = controlled_lights[i % len(controlled_lights)]

        # Create an environment where the ith light in controlled_lights is being trained
        env = create_env(learning_light['light_name'], num_proc, steps_per_episode, global_config_params['visualize_training'])

        # Load existing model for the learning light if it exists
        path_name = f'Scenarios/{config_params["model_save_path"]}/PPO2_{learning_light["light_name"]}'
        if global_config_params['use_tensorboard']:
            if os.path.isfile(path_name + '.zip'):
                model = PPO2.load(path_name, env=env, tensorboard_log=f'./Scenarios/{config_params["model_save_path"]}/tensorboard/{learning_light["light_name"]}/')
            else:
                model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=f'./Scenarios/{config_params["model_save_path"]}/tensorboard/{learning_light["light_name"]}/')
        else:
            if os.path.isfile(path_name + '.zip'):
                model = PPO2.load(path_name, env=env)
            else:
                model = PPO2(MlpPolicy, env, verbose=1)

        train_start_time = time.time()
        model.learn(total_timesteps=steps_per_episode * num_episodes)
        print(f'LIGHT LEARNING TIME: {time.time() - train_start_time}')
        model.save(f'Scenarios/{config_params["model_save_path"]}/PPO2_{learning_light["light_name"]}')
        print(f'DONE LEARNING LIGHT: {learning_light["light_name"]}')
        env.close()
        del env


# Create a sumo environment
def create_env(node_name, num_proc, steps_per_episode, visualize_training):
    if num_proc == 1:
        env = DummyVecEnv([lambda: SumoEnvParallel(steps_per_episode, visualize_training, node_name)])
    else:
        if sys.platform == 'win32':
            thread_method = 'spawn'
        else:
            thread_method = 'forkserver'  # fork was having issues with multi-agent for me so I switched to forkserver
        env = SubprocVecEnv([lambda: SumoEnvParallel(steps_per_episode, visualize_training, node_name) for _ in range(num_proc)],
                            start_method=thread_method)
    return env


if __name__ == '__main__':
    main()
