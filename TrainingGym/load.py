# Remove TF warnings in Stable baselines (may not be safe)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env.SumoEnvParallel import SumoEnvParallel
import json

with open('global_config.json') as global_json_file:
    local_config_path = json.load(global_json_file)['config_path']
with open(f'Scenarios/{local_config_path}') as json_file:
    config_params = json.load(json_file)

parallel = config_params['num_proc'] > 1

steps_per_episode = config_params["steps_per_episode"]
num_episodes = config_params["num_episodes"]

env = DummyVecEnv([lambda: SumoEnvParallel(steps_per_episode, True)])

model = PPO2.load(f'Scenarios/{config_params["model_save_path"]}')
obs = env.reset()
total_rewards = 0
for i in range(steps_per_episode):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    total_rewards += rewards[0]
    if done:
        break
print(total_rewards)
