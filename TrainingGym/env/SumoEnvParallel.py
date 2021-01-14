from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import gym
from gym import spaces, logger
from stable_baselines import PPO2
from stable_baselines.common.callbacks import BaseCallback
import numpy as np
import json

# we need to import python modules from the $SUMO_HOME/tools directory
# TODO: This line isn't necessary if everyone directly installs the sumo python libraries in their python dist
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary

with open('global_config.json') as global_json_file:
    local_config_path = json.load(global_json_file)['config_path']
with open(f'Scenarios/{local_config_path}') as json_file:
    config_params = json.load(json_file)

# Load config
controlled_lights = config_params['controlled_lights']
uncontrolled_lights = config_params['uncontrolled_lights']
all_important_roads = set()
for i_node in controlled_lights:
    for i_road in i_node['important_roads']:
        all_important_roads.add(i_road)
load_options = ["-c", f'Scenarios/{config_params["sumocfg_path"]}', "--tripinfo-output",
                f'Scenarios/{config_params["tripinfo_output_path"]}', "-t"]


# Find an object with a given value for an attribute in a list
def find_attr_in_list(lst, attr, value):
    for obj in lst:
        if obj[attr] == value:
            return obj
    return None


class SumoEnvParallel(gym.Env, BaseCallback):
    def __init__(self, steps_per_episode, validation_env, controlled_node_name):
        super(SumoEnvParallel, self).__init__()

        # Environment parameters
        self.steps_per_episode = steps_per_episode
        self.is_done = False
        self.current_step = 0

        # Get the node which this agent is controlling
        self.controlled_node = find_attr_in_list(controlled_lights, 'name', controlled_node_name)

        # Setup action, reward, and observation spaces
        self.reward_range = (-float('inf'), float('inf'))  # TODO: Restrict this to our final reward function
        self.action_space = spaces.Discrete(self.controlled_node['num_phases'])
        self.observation_space = spaces.Box(low=0, high=float('inf'),
                                            shape=np.array([len(self.controlled_node['important_roads']) * 2]),
                                            dtype=np.float32)

        # Start connection with sumo
        import traci  # each gym environment instance has a discrete traci instance
        self.sumo = traci
        self.sumoBinary = checkBinary('sumo-gui') if validation_env else checkBinary('sumo')
        self.sumo_started = False

        # Get existing models for controlled, but not learning lights
        self.model_list = []
        for node in controlled_lights:
            if node['name'] != controlled_node_name:
                model_path = f'Scenarios/{config_params["model_save_path"]}/PPO2_{node["name"]}'
                if os.path.isfile(model_path + '.zip'):
                    self.model_list.append({'node': node, 'model': PPO2.load(model_path), 'next_phase': 0})
                else:
                    self.model_list.append({'node': node, 'model': None, 'next_phase': 0})

    def reset(self):
        # Sumo is started on the first call to reset
        if not self.sumo_started:
            self.sumo.start([self.sumoBinary] + load_options)
            self.sumo_started = True
        # Sumo should be started on subsequent resets
        else:
            self.sumo.load(load_options)
        self.current_step = 0
        self.is_done = False
        return self._next_observation(self.controlled_node)

    def step(self, action):
        # Determine next phase for controlled lights not learning
        for model in self.model_list:
            if model['model'] is not None:
                other_obs = self._next_observation(model['node'])
                model['next_phase'] = model['model'].predict(other_obs)[0]
            # If no model exists yet just keep it on current phase (This will only be the case for the
            #  first time training a set of lights)

        # Set all controlled light phases
        self._take_action(action)

        # Advance simulation
        self.sumo.simulationStep()
        self.current_step += 1

        # Get obs and reward
        obs = self._next_observation(self.controlled_node)
        reward = self._get_reward()

        # Check if the simulation is already done (it shouldn't be)
        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                        "You should always call 'reset()' once you receive 'done = True' "
                        "-- any further steps are undefined behavior.")
            reward = 0.0

        # If the next step of the simulation is the last step of the episode, indicate the episode is done
        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        return obs, reward, self.is_done, {}

    def _next_observation(self, node):
        # TODO: Consider a different observation function
        obs = []
        wait_counts, road_counts = self._get_road_waiting_vehicle_count()
        # For all important roads to this node add their vehicle and waiting vehicle counts
        for road in node['important_roads']:
            if road in road_counts.keys():
                obs.append(road_counts[road])
                obs.append(wait_counts[road])
            else:
                obs.append(0)
                obs.append(0)
        return np.array(obs)

    def _get_reward(self):
        # TODO: Consider a different reward function
        road_waiting_vehicles_dict, _ = self._get_road_waiting_vehicle_count()
        reward = 0.0
        for (road_id, num_vehicles) in road_waiting_vehicles_dict.items():
            if road_id in all_important_roads:
                reward -= num_vehicles
        return reward

    def _take_action(self, action):
        # Change phase for learning light
        if action != self.controlled_node['curr_phase']:
            self.controlled_node['curr_phase'] = action
            self._set_tl_phase(self.controlled_node['name'], action)

        # Change phase for controlled lights not currently learning
        for model in self.model_list:
            if model['next_phase'] != model['node']['curr_phase']:
                model['node']['curr_phase'] = model['next_phase']
                self._set_tl_phase(model['node']['name'], model['next_phase'])

    def _get_road_waiting_vehicle_count(self):
        # TODO: Find a more efficient way of getting these values (SUMO has a batch data function that might be
        #  interesting)
        wait_counts = {}
        road_counts = {}
        vehicles = self.sumo.vehicle.getIDList()
        for v in vehicles:
            road = self.sumo.vehicle.getRoadID(v)
            if road in all_important_roads:
                if road not in wait_counts.keys():
                    wait_counts[road] = 0
                    road_counts[road] = 0
                if self.sumo.vehicle.getWaitingTime(v) > 0:
                    wait_counts[road] += 1
                road_counts[road] += 1
        return wait_counts, road_counts

    def _set_tl_phase(self, intersection_id, phase_id):
        self.sumo.trafficlight.setPhase(intersection_id, phase_id)

    def close(self):
        print('Closing SUMO')
        self.sumo.close()
        self.sumo_started = False
        del self.model_list
        self.model_list = []

    def render(self, **kwargs):
        # Rendering is set when the environment is initialized with the validation_env flag
        pass
