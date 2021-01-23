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
from traci._trafficlight import Phase, Logic

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

# Constants
YELLOW_LENGTH = 4.2  # seconds
RED_LENGTH = 1.8  # seconds
STEP_LENGTH = 1.0  # seconds

# Find an object with a given value for an attribute in a list
def find_attr_in_list(lst, attr, value):
    for obj in lst:
        if obj[attr] == value:
            return obj
    return None


class SumoEnvParallel(gym.Env, BaseCallback):
    def __init__(self, steps_per_episode, testing_env, controlled_light_name):
        super(SumoEnvParallel, self).__init__()

        # Environment parameters
        self.steps_per_episode = steps_per_episode
        self.is_done = False
        self.current_step = 0

        # Get the node which this agent is controlling
        self.controlled_node = find_attr_in_list(controlled_lights, 'light_name', controlled_light_name)

        # Setup action, reward, and observation spaces
        self.action_space = spaces.Discrete(len(self.controlled_node['states']))
        self.observation_space = spaces.Box(low=0, high=float('inf'),
                                            shape=np.array([len(self.controlled_node['important_roads']) * 3 + 2]),
                                            dtype=np.float32)

        # Start connection with sumo
        import traci  # each gym environment instance has a discrete traci instance
        self.sumo = traci
        self.sumoBinary = checkBinary('sumo-gui') if testing_env else checkBinary('sumo')
        self.sumo_started = False

        # Get existing models for controlled, but not learning lights
        self.model_list = []
        for node in controlled_lights:
            if node['light_name'] != controlled_light_name:
                model_path = f'Scenarios/{config_params["model_save_path"]}/PPO2_{node["light_name"]}'
                if os.path.isfile(model_path + '.zip'):
                    self.model_list.append({'node': node, 'model': PPO2.load(model_path), 'next_phase': 0})
                else:
                    self.model_list.append({'node': node, 'model': None, 'next_phase': 0})

        # Michael's added stuff TODO: Review if needed
        self.total_reward = 0
        self.current_action = 0
        self.previous_action = 0
        self.action_time = 0
        self.proximity = 30  # later turn this into an env parameter that can be changed

    def reset(self):
        # Sumo is started on the first call to reset
        if not self.sumo_started:
            self.sumo.start([self.sumoBinary] + load_options)
            self.sumo_started = True
        # Sumo should be started on subsequent resets
        else:
            self.sumo.load(load_options + ["--start"])
        self.current_step = 0
        self.is_done = False

        # Michael's new stuff TODO: review these
        self.total_reward = 0
        self.current_action = 0
        self.previous_action = 0
        self.action_time = 0
        return self._next_observation(self.controlled_node)

    def step(self, action):
        self.previous_action = self.current_action
        self.current_action = action
        # Determine next phase for controlled lights not learning
        for model in self.model_list:
            if model['model'] is not None:
                other_obs = self._next_observation(model['node'])
                model['next_phase'] = model['model'].predict(other_obs)[0]
            # If no model exists yet just keep it on current phase (This will only be the case for the
            #  first time training a set of lights)

        # Set all controlled light phases
        self._take_action(action)

        # Update phase step counts
        for node in controlled_lights:
            node['steps_since_last_change'] += 1

        # Advance simulation
        self.sumo.simulationStep()
        self.current_step += 1

        # Get obs and reward
        obs = self._next_observation(self.controlled_node)
        reward = self._get_reward()
        self.total_reward += reward

        # If the next step of the simulation is the last step of the episode, indicate the episode is done
        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True
        return obs, reward, self.is_done, {}

    def _next_observation(self, node):
        # TODO: Consider a different observation function
        obs = []
        wait_counts, far_counts, near_counts = self._get_road_waiting_vehicle_count(node)
        # For all important roads to this node add their vehicle and waiting vehicle counts
        for road in node['important_roads']:
            if road in far_counts.keys():
                obs.append(far_counts[road])
                obs.append(wait_counts[road])
                obs.append(near_counts[road])
            else:
                obs.append(0)
                obs.append(0)
                obs.append(0)
        # obs.append(self.action_time)
        obs.append(self.controlled_node['steps_since_last_change'])
        obs.append(self.controlled_node['last_phase'])
        # obs.append(self.previous_action)
        return np.array(obs)

    def _get_reward(self):
        # TODO: Consider a different reward function
        road_waiting_vehicles_dict, _, _ = self._get_road_waiting_vehicle_count(self.controlled_node)
        reward = 0.0
        if self.current_action != self.previous_action:
            reward -= 1

        for (road_id, vehicle_wait_time) in road_waiting_vehicles_dict.items():
            if road_id in self.controlled_node['important_roads']:
                reward -= vehicle_wait_time

        return reward

    def _take_action(self, action):
        # Change phase for learning light
        self._update_tl(self.controlled_node, action)

        # Change phase for controlled lights not currently learning
        for model in self.model_list:
            self._update_tl(model['node'], model['next_phase'])

    def _update_tl(self, node, next_phase):
        # Make sure the light isn't in a a yellow or red phase
        if node['steps_since_last_change'] * STEP_LENGTH > (RED_LENGTH + YELLOW_LENGTH):

            # New phase, create a new Logic for it and reset counter
            if next_phase != node['curr_phase']:
                node['steps_since_last_change'] = 0  # reset counter
                node['last_phase'] = node['curr_phase']
                node['curr_phase'] = next_phase

                # Create state strings for yellow, red, and green phases
                yellow_string = list(node['states'][node['last_phase']])
                red_string = list(node['states'][node['last_phase']])
                green_string = node['states'][node['curr_phase']]
                for i_char in range(len(green_string)):
                    # If any green becomes red, or a green major becomes minor we need yellow and red phases for
                    # those connections
                    if yellow_string[i_char].lower() == 'g' and green_string[i_char] == 'r' or yellow_string[i_char] == 'G' and green_string[i_char] == 'g':
                        yellow_string[i_char] = 'y'
                        red_string[i_char] = 'r'
                yellow_string = ''.join(yellow_string)
                red_string = ''.join(red_string)

                # Create Phase and Logic objects to send to SUMO
                phases = [Phase(YELLOW_LENGTH, yellow_string), Phase(RED_LENGTH, red_string), Phase(9999, green_string)]
                logic = Logic("1", 0, 0, phases=phases)
                self._set_tl_logic(node['light_name'], logic)
                self.sumo.trafficlight.setProgram(node['light_name'], "1")
            # Set light duration again, counter continues
            else:
                self._set_tl_ryg(node['light_name'], node['states'][node['curr_phase']])

    def _get_road_waiting_vehicle_count(self, node):
        # TODO: Find a more efficient way of getting these values (SUMO has a batch data function that might be
        #  interesting)
        wait_counts = {}
        far_counts = {}
        near_counts = {}
        junc_x, junc_y = self.sumo.junction.getPosition(node['node_name'])
        vehicles = self.sumo.vehicle.getIDList()
        for v in vehicles:
            road = self.sumo.vehicle.getRoadID(v)
            v_x, v_y = (self.sumo.vehicle.getPosition(v))
            if road in node['important_roads']:
                if road not in wait_counts.keys():
                    wait_counts[road] = 0
                    far_counts[road] = 0
                    near_counts[road] = 0
                if self.sumo.vehicle.getWaitingTime(v) > 0:
                    wait_counts[road] += self.sumo.vehicle.getWaitingTime(v)
                if self.sumo.simulation.getDistance2D(junc_x, junc_y, v_x, v_y) <= self.proximity:
                    near_counts[road] += 1
                else:
                    far_counts[road] += 1
        return wait_counts, far_counts, near_counts

    def _set_tl_logic(self, light_id, logic):
        self.sumo.trafficlight.setProgramLogic(light_id, logic)

    def _set_tl_phase(self, light_id, phase_id):
        self.sumo.trafficlight.setPhase(light_id, phase_id)

    def _set_tl_ryg(self, light_id, ryg_string):
        self.sumo.trafficlight.setRedYellowGreenState(light_id, ryg_string)

    def close(self):
        print('Closing SUMO')
        self.sumo.close()
        self.sumo_started = False
        del self.model_list
        self.model_list = []

    def render(self, **kwargs):
        # Rendering is set when the environment is initialized with the validation_env flag
        pass
