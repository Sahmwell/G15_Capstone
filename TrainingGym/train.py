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
import json
import os
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict
import numpy as np
import hashlib


def main():

    # Set subprocess start method for each light's training set
    thread_method = 'spawn'  # fork was having issues with multi-agent for me so I switched to forkserver
    mp.set_start_method(thread_method)

    # Load configs
    with open('global_config.json') as global_json_file:
        global_config_params = json.load(global_json_file)
        local_config_path = global_config_params['config_path']
    with open(f'Scenarios/{local_config_path}') as json_file:
        config_params = json.load(json_file)

    # Make sure that training_models, stats, and tensorboard folders exist for this scenario
    os.makedirs(os.path.join('Scenarios', config_params['model_save_path'], 'stats'), exist_ok=True)
    os.makedirs(os.path.join('Scenarios', config_params['model_save_path'], 'tensorboard'), exist_ok=True)

    # Load Config Parameters
    controlled_lights = config_params['controlled_lights']
    for i in range(len(controlled_lights) - 1, -1, -1):
        if not controlled_lights[i]['train']:
            del controlled_lights[i]
    num_trials = config_params['num_trials']
    num_workers_per_light = global_config_params['num_proc'] // len(controlled_lights)

    # Statistics list (will be saved as a numpy object later)
    stats = []

    # Run a learning session on each light
    start_datetime = datetime.now()
    log_to_file(f'BEGINNING {num_trials} TRIAL TRAINING LOOP AT {start_datetime}')
    loop_start_time = time.time()
    for i_trial in range(num_trials):
        log_to_file(f'    Starting trial {i_trial}')
        trial_start_time = time.time()
        light_procs = []
        for learning_light in controlled_lights:
            p = mp.Process(target=learn_light, args=(learning_light, num_workers_per_light, global_config_params, config_params, i_trial))
            p.start()
            light_procs.append(p)
        for proc in light_procs:
            proc.join()
        log_to_file(f'    Done trial {i_trial} in {time.time() - trial_start_time} s (Elapsed time {datetime.now() - start_datetime})')

        # Collect statistics
        if i_trial % global_config_params['trials_per_statistic_gather'] == 0:
            print(f'Collecting statistics in {global_config_params["statistic_episodes"]} episodes ...')
            start_stats = time.time()
            next_stats = collect_statistics(controlled_lights, global_config_params["statistic_episodes"], config_params)
            stats.append([next_stats[node['node_name']] for node in controlled_lights])
            stats_path = os.path.join('.', 'Scenarios', config_params["model_save_path"], 'stats', hashlib.sha1(str(loop_start_time).encode()).hexdigest() + '.npy')
            np.save(stats_path, np.array(stats))
            log_to_file(f'    Collected statistics after trial {i_trial} saved in {stats_path} (Took {time.time() - start_stats} s)')

    log_to_file(f'FINISHED TRAINING LOOP in {time.time() - loop_start_time} s')


def learn_light(learning_light, num_workers, global_config_params, config_params, i_trial):

    # Create an environment where the ith light in controlled_lights is being trained
    sumo_gym = create_env(learning_light['light_name'], num_workers, config_params['steps_per_episode'],
                     global_config_params['visualize_training'], config_params['training_seed'])

    # Load existing model for the learning light if it exists
    model_path = f'Scenarios/{config_params["model_save_path"]}/PPO2_{learning_light["light_name"]}'
    if global_config_params['use_tensorboard']:
        if os.path.isfile(model_path + '.zip'):
            model = PPO2.load(model_path, env=sumo_gym,
                              tensorboard_log=f'./Scenarios/{config_params["model_save_path"]}/tensorboard/{learning_light["light_name"]}/')
        else:
            model = PPO2(MlpPolicy, sumo_gym, verbose=1,
                         tensorboard_log=f'./Scenarios/{config_params["model_save_path"]}/tensorboard/{learning_light["light_name"]}/')
    else:
        if os.path.isfile(model_path + '.zip'):
            model = PPO2.load(model_path, env=sumo_gym)
        else:
            model = PPO2(MlpPolicy, sumo_gym, verbose=1)

    train_start_time = time.time()
    model.learn(total_timesteps=config_params['steps_per_episode'] * config_params['num_episodes'])
    print(f'LIGHT LEARNING TIME: {time.time() - train_start_time}')
    model.save(f'Scenarios/{config_params["model_save_path"]}/PPO2_{learning_light["light_name"]}')
    if i_trial % global_config_params['trials_per_checkpoint'] == 0:
        model.save(f'Scenarios/{config_params["model_save_path"]}/PPO2_{learning_light["light_name"]}_{i_trial}')
    print(f'DONE LEARNING LIGHT: {learning_light["light_name"]}')
    sumo_gym.close()


def collect_statistics(controlled_lights, num_episodes, config_params):
    # Create sumo environment
    sumo_env = SumoEnvParallel(config_params['steps_per_episode'], False, controlled_lights[0]['light_name'], collect_statistics=True)

    # Load each light's model
    model = PPO2.load(f'Scenarios/{config_params["model_save_path"]}/PPO2_{controlled_lights[0]["light_name"]}')

    # Reset and run the environment
    total_rewards = defaultdict(lambda: 0)
    for _ in range(num_episodes):
        done = False
        obs = sumo_env.reset()
        while not done:
            action, state = model.predict(obs)
            # Since we're not training, it doesn't matter which light is the first parameter
            obs, rewards, done, info = sumo_env.step(action)
            for node in info['statistics']:
                total_rewards[node['node_name']] += node['step_reward']

    for node in total_rewards.keys():
        total_rewards[node] /= num_episodes

    sumo_env.close()
    del model

    return total_rewards


# Create a sumo environment
def create_env(node_name, num_proc, steps_per_episode, visualize_training, seed):
    if num_proc == 1:
        return DummyVecEnv([lambda: SumoEnvParallel(steps_per_episode, visualize_training, node_name, collect_statistics=False, seed=seed)])
    else:
        thread_method = 'spawn'  # fork was having issues with multi-agent for me so I switched to spawn
        return SubprocVecEnv([lambda: SumoEnvParallel(steps_per_episode, visualize_training, node_name, collect_statistics=False, seed=seed) for _ in range(num_proc)],
                            start_method=thread_method)


def log_to_file(msg):
    with open('training_log.txt', 'a') as f:
        f.write(msg + '\n')


if __name__ == '__main__':
    main()
