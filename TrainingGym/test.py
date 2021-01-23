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

# Get config parameters
steps_per_episode = config_params['test_steps']
num_episodes = config_params["num_episodes"]
controlled_lights = config_params['controlled_lights']

# Create sumo environment
env = SumoEnvParallel(steps_per_episode, True, controlled_lights[0]['light_name'])

# Load each light's model
model = PPO2.load(f'Scenarios/{config_params["model_save_path"]}/PPO2_{controlled_lights[0]["light_name"]}')

# Reset and run the environment
obs = env.reset()
total_rewards = 0  # Count the sum of the reward function over all time steps
for i in range(config_params['test_steps']):
    action, state = model.predict(obs)
    # Since we're not training, it doesn't matter which light is the first parameter
    obs, rewards, done, info = env.step(action)
    total_rewards += rewards
    if done:
        break
print(total_rewards)
