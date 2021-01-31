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
import multiprocessing as mp
from datetime import datetime


def main():
    # Load configs
    with open('global_config.json') as global_json_file:
        global_config_params = json.load(global_json_file)
        local_config_path = global_config_params['config_path']
    with open(f'Scenarios/{local_config_path}') as json_file:
        config_params = json.load(json_file)

    # Load Config Parameters
    controlled_lights = config_params['controlled_lights']
    for i in range(len(controlled_lights) - 1, -1, -1):
        if not controlled_lights[i]['train']:
            del controlled_lights[i]
    num_trials = config_params['num_trials']
    num_workers_per_light = global_config_params['num_proc'] // len(controlled_lights)

    # Run a learning session on each light
    log_to_file(f'BEGINNING {num_trials} TRIAL TRAINING LOOP AT {datetime.now()}')
    loop_start_time = time.time()
    for i_trial in range(num_trials):
        log_to_file(f'    Starting trial {i_trial}')
        trial_start_time = time.time()
        light_procs = []
        for i_light in range(len(controlled_lights)):
            learning_light = controlled_lights[i_light]
            p = mp.Process(target=learn_light, args=(learning_light, num_workers_per_light, global_config_params, config_params))
            p.start()
            light_procs.append(p)
        for proc in light_procs:
            proc.join()
        log_to_file(f'    Done trial {i_trial} in {time.time() - trial_start_time} s')
    log_to_file(f'FINISHED TRAINING LOOP in {time.time() - loop_start_time} s')


def learn_light(learning_light, num_workers, global_config_params, config_params):

    # Create an environment where the ith light in controlled_lights is being trained
    env = create_env(learning_light['light_name'], num_workers, config_params['steps_per_episode'],
                     global_config_params['visualize_training'])

    # Load existing model for the learning light if it exists
    model_path = f'Scenarios/{config_params["model_save_path"]}/PPO2_{learning_light["light_name"]}'
    if global_config_params['use_tensorboard']:
        if os.path.isfile(model_path + '.zip'):
            model = PPO2.load(model_path, env=env,
                              tensorboard_log=f'./Scenarios/{config_params["model_save_path"]}/tensorboard/{learning_light["light_name"]}/')
        else:
            model = PPO2(MlpPolicy, env, verbose=1,
                         tensorboard_log=f'./Scenarios/{config_params["model_save_path"]}/tensorboard/{learning_light["light_name"]}/')
    else:
        if os.path.isfile(model_path + '.zip'):
            model = PPO2.load(model_path, env=env)
        else:
            model = PPO2(MlpPolicy, env, verbose=1)

    train_start_time = time.time()
    model.learn(total_timesteps=config_params['steps_per_episode'] * config_params['num_episodes'])
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


def log_to_file(msg):
    with open('training_log.txt', 'a') as f:
        f.write(msg + '\n')


if __name__ == '__main__':
    main()
