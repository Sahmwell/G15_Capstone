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
import time

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

controlled_lights = config_params['controlled_lights']
uncontrolled_lights = config_params['uncontrolled_lights']
# important_roads = config_params['important_roads']
all_important_roads = set()
for i_node in controlled_lights:
    for road in i_node['important_roads']:
        all_important_roads.add(road)
load_options = ["-c", f'Scenarios/{config_params["sumocfg_path"]}', "--tripinfo-output",
                f'Scenarios/{config_params["tripinfo_output_path"]}', "-t"]


def find_attr_in_list(lst, attr, value):
    for obj in lst:
        if obj[attr] == value:
            return obj
    return None


class SumoEnvParallel(gym.Env, BaseCallback):
    def __init__(self, steps_per_episode, validation_env, controlled_node_name):
        super(SumoEnvParallel, self).__init__()
        # Environment parameters
        # self.scenario_name = scenario_name
        self.steps_per_episode = steps_per_episode
        self.is_done = False
        self.current_step = 0

        # Get the node which this agent is controlling
        self.controlled_node = find_attr_in_list(controlled_lights, 'name', controlled_node_name)

        # Setup action, reward, and observation spaces
        self.reward_range = (-float('inf'), float('inf'))  # HARDCODE
        self.action_space = spaces.Discrete(controlled_lights[0]['num_phases'])  # HARDCODE
        # self.action_space = spaces.Box(low=0, high=controlled_lights[0]['num_phases'] - 1, shape=np.array([2]), dtype=np.int)
        self.observation_space = spaces.Box(low=0, high=float('inf'),
                                            shape=np.array([len(self.controlled_node['important_roads']) * 2]),
                                            dtype=np.float32)

        # Start connection with sumo
        import traci  # each gym environment instance has a discrete traci instance
        self.sumo = traci
        self.sumoBinary = checkBinary('sumo-gui') if validation_env else checkBinary('sumo')
        self.sumo_started = False

        # If we're training get existing models for other lights and use them to switch the other lights
        self.model_list = []
        for node in controlled_lights:
            if node['name'] != controlled_node_name:
                path_name = f'Scenarios/{config_params["model_save_path"]}/PPO2_{node["name"]}'
                if os.path.isfile(path_name + '.zip'):
                    self.model_list.append({'node_name': node['name'], 'model': PPO2.load(path_name), 'next_phase': 0})
                else:
                    self.model_list.append({'node_name': node['name'], 'model': None, 'next_phase': 0})

    def reset(self):
        if not self.sumo_started:
            self.sumo.start([self.sumoBinary] + load_options)
            self.sumo_started = True
        else:
            self.sumo.load(load_options)
        self.current_step = 0
        self.is_done = False
        return self._next_observation(self.controlled_node)

    def step(self, action):

        # Determine next phase for lights not being learned
        for model in self.model_list:
            if model['model'] is not None:
                node = find_attr_in_list(controlled_lights, 'name', model['node_name'])
                other_obs = self._next_observation(node)
                model['next_phase'] = model['model'].predict(other_obs)[0]
            # If no model exists yet just keep it on current phase

        self._take_action(action)

        self.sumo.simulationStep()
        self.current_step += 1

        obs = self._next_observation(self.controlled_node)
        reward = self._get_reward()

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                        "You should always call 'reset()' once you receive 'done = True' "
                        "-- any further steps are undefined behavior.")
            reward = 0.0

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        return obs, reward, self.is_done, {}

    def _next_observation(self, node):
        obs = []
        wait_counts, road_counts = self._get_road_waiting_vehicle_count()
        # HARDCODE
        # node = controlled_lights[controlled_lights['name'] == node_name]
        for road in node['important_roads']:
            if road in road_counts.keys():
                obs.append(road_counts[road])
                obs.append(wait_counts[road])
            else:
                obs.append(0)
                obs.append(0)

        return np.array(obs)

    def _get_reward(self):
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

        # Change phase for lights not currently learning
        for model in self.model_list:
            other_node = find_attr_in_list(controlled_lights, 'name', model['node_name'])
            if model['next_phase'] != other_node['curr_phase']:
                other_node['curr_phase'] = model['next_phase']
                self._set_tl_phase(other_node['name'], model['next_phase'])

    def _get_road_waiting_vehicle_count(self):
        wait_counts = {}
        road_counts = {}
        vehicles = self.sumo.vehicle.getIDList()
        for v in vehicles:
            road = self.sumo.vehicle.getRoadID(v)
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


