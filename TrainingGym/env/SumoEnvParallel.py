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
from collections import defaultdict

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
    global_config_params = json.load(global_json_file)
    local_config_path = global_config_params['config_path']
with open(f'Scenarios/{local_config_path}') as json_file:
    config_params = json.load(json_file)

# Constants
YELLOW_LENGTH = 4.2  # seconds
RED_LENGTH = 1.8  # seconds
STEP_LENGTH = 1.0  # seconds

# Load config
controlled_lights = config_params['controlled_lights']
for i in range(len(controlled_lights) - 1, -1, -1):
    if not controlled_lights[i]['train']:
        del controlled_lights[i]
all_important_roads = set()
for i_node in controlled_lights:
    for i_direction in i_node['connections']:
        for i_edge in i_direction['edges']:
            all_important_roads.add(i_edge)
load_options = ["-c", f'Scenarios/{config_params["sumocfg_path"]}', "--start", "--quit-on-end", "--step-length", str(STEP_LENGTH), "--random", "--no-warnings", "true"]


# Find an object with a given value for an attribute in a list
def find_attr_in_list(lst, attr, value):
    for obj in lst:
        if obj[attr] == value:
            return obj
    return None


class SumoEnvParallel(gym.Env, BaseCallback):
    def __init__(self, steps_per_episode, use_sumo_gui, controlled_light_name, collect_statistics):
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
                                            shape=np.array([len(self.controlled_node['connections']) * 3 + 2]),
                                            dtype=np.float32)

        # Start connection with sumo
        import traci  # each gym environment instance has a discrete traci instance
        self.sumo = traci
        self.sumoBinary = checkBinary('sumo-gui') if use_sumo_gui else checkBinary('sumo')
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

        # Variables to hold sumo state values
        self.vehicles_on_edge = defaultdict(lambda: [])  # dict where key is the road name, and value is a list of
        # cars with the members: name, wait_time, pos

        # Michael's added stuff TODO: Review if needed
        self.total_reward = 0
        self.current_action = 0
        self.previous_action = 0
        self.action_time = 0
        self.near_dist = config_params["near_distance"]  # TODO: Turn this into an env parameter that can be changed
        self.far_dist = config_params['far_distance']  # TODO: Review this number. It was randomly selected

        self.collect_statistics = collect_statistics

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

        # No vehicles in the simulation yet
        self.vehicles_on_edge = defaultdict(lambda: [])

        # Michael's new stuff TODO: review these
        self.sumo.poi.add('poi_0', -100, 200, (255, 0, 0), poiType='test')
        self.total_reward = 0
        self.current_action = 0
        self.previous_action = 0
        self.action_time = 0
        return self._next_observation(self.controlled_node)

    def step(self, action):
        info = {}
        self.previous_action = self.current_action
        self.current_action = action
        # Determine next phase for controlled lights not learning
        for model in self.model_list:
            if model['model']:
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

        # Get sumo state information
        self.vehicles_on_edge = self._get_sumo_values()  # This data is used by the observation returned from
        # the step function as well as the observation for each agent at the start of the next call to step

        # Get obs and reward
        obs = self._next_observation(self.controlled_node)
        reward = self._get_reward(self.controlled_node)
        self.total_reward += reward

        # If the next step of the simulation is the last step of the episode, indicate the episode is done
        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        if self.collect_statistics:
            statistics = [{'node_name': self.controlled_node['node_name'], 'step_reward':reward}]
            for model in self.model_list:
                statistics.append({'node_name': model['node']['node_name'], 'step_reward': self._get_reward(model['node'])})
            info['statistics'] = statistics

        self.sumo.poi.setType('poi_0', str(action) + ", " + str(reward) + ", " + str(self.total_reward))
        return obs, reward, self.is_done, info

    # Retrieve values from sumo for the current time step
    def _get_sumo_values(self):
        vehicles_on_edge = defaultdict(lambda: [])
        for edge in all_important_roads:
            vehicles = self.sumo.edge.getLastStepVehicleIDs(edge)
            for v in vehicles:
                v_pos = np.array(self.sumo.vehicle.getPosition(v))
                vehicles_on_edge[edge].append(
                    {"name": v, "wait_time": self.sumo.vehicle.getWaitingTime(v), "pos": v_pos}
                )
        return vehicles_on_edge

    def _next_observation(self, node):
        obs = []
        total_wait_times, far_vehicle_count, near_vehicle_count = self._get_direction_vehicle_counts(node)
        # For all incoming directions to this junction add their metrics to the observation
        for direction in node['connections']:
            obs.append(total_wait_times[direction['label']])
            obs.append(far_vehicle_count[direction['label']])
            obs.append(near_vehicle_count[direction['label']])
        # obs.append(self.action_time)
        obs.append(self.controlled_node['steps_since_last_change'])
        obs.append(self.controlled_node['last_phase'])
        # obs.append(self.previous_action)
        return np.array(obs)

    def _get_reward(self, node):
        road_waiting_vehicles_dict, _, _ = self._get_direction_vehicle_counts(node)
        reward = 0.0
        if self.current_action != self.previous_action:
            reward -= 5

        for direction in node['connections']:
            reward -= road_waiting_vehicles_dict[direction['label']]

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
                # TODO: We could speed this up with a dynamic programming approach (use a dict to hold these intermediates)
                yellow_string = list(node['states'][node['last_phase']])
                red_string = list(node['states'][node['last_phase']])
                green_string = node['states'][node['curr_phase']]
                for i_char in range(len(green_string)):
                    # If any green becomes red, or a green major becomes minor we need yellow and red phases for
                    # those connections
                    if (yellow_string[i_char].lower() == 'g' and (green_string[i_char].lower() in ['r', 's'])) or yellow_string[i_char] == 'G' and green_string[i_char] == 'g':
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

    def _get_direction_vehicle_counts(self, node):
        # This function assumes that self._get_sumo_values() has been called for the current timestep
        total_wait_times = defaultdict(lambda: 0)
        far_vehicle_count = defaultdict(lambda: 0)
        near_vehicle_count = defaultdict(lambda: 0)
        junc_pos = np.array(self.sumo.junction.getPosition(node['node_name']))
        for direction in node['connections']:
            for edge in direction['edges']:
                for vehicle in self.vehicles_on_edge[edge]:
                    total_wait_times[direction['label']] += vehicle['wait_time']
                    dist_to_junc = np.linalg.norm(vehicle['pos'] - junc_pos)
                    if dist_to_junc <= self.near_dist:
                        near_vehicle_count[direction['label']] += 1
                    elif dist_to_junc <= self.far_dist:
                        far_vehicle_count[direction['label']] += 1
        return total_wait_times, far_vehicle_count, near_vehicle_count

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
