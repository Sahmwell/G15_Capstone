from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import gym
from gym import spaces, logger
import numpy as np

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import traci


controlled_lights = [{'name': 'gneJ14', 'curr_phase': 0, 'num_phases': 6}]
uncontrolled_lights = [{'name': 'nw', 'curr_phase': 0, 'num_phases': 4},
                       {'name': 'se', 'curr_phase': 0, 'num_phases': 4},
                       {'name': 'sw', 'curr_phase': 0, 'num_phases': 4}]
important_roads = ['gneE16', 'gneE59', 'gneE13']
load_options = ["-c", "PoundSign/PoundSign.sumocfg", "--tripinfo-output", "PoundSign/tripinfo.xml", "-t"]


class SumoEnv(gym.Env):
    def __init__(self, steps_per_episode, render):
        super(SumoEnv, self).__init__()
        # self.scenario_name = scenario_name
        self.steps_per_episode = steps_per_episode
        self.is_done = False
        self.current_step = 0

        self.reward_range = (-float('inf'), float('inf'))  # HARDCODE
        self.action_space = spaces.Discrete(5)  # HARDCODE
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=np.array([6]), dtype=np.float32)  # HARDCODE

        # Start connection with sumo
        self.noguiBinary = checkBinary('sumo')
        self.guiBinary = checkBinary('sumo-gui')
        # self.current_binary = self.noguiBinary
        self.current_binary = self.guiBinary if render else self.noguiBinary
        traci.start([self.current_binary] + load_options)

    def reset(self):
        traci.load(load_options)
        self.current_step = 0
        self.is_done = False

        return self._next_observation()

    def _next_observation(self):
        obs = []
        wait_counts, road_counts = self._get_road_waiting_vehicle_count()
        # HARDCODE
        for lane in important_roads:
            obs.append(road_counts[lane])
            obs.append(wait_counts[lane])

        return np.array(obs)

    def step(self, action):

        self._take_action(action)

        traci.simulationStep()
        self.current_step += 1

        obs = self._next_observation()
        reward = self._get_reward()

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                        "You should always call 'reset()' once you receive 'done = True' "
                        "-- any further steps are undefined behavior.")
            reward = 0.0

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        return obs, reward, self.is_done, {}

    def _get_reward(self):
        road_waiting_vehicles_dict, _ = self._get_road_waiting_vehicle_count()
        reward = 0.0

        for (road_id, num_vehicles) in road_waiting_vehicles_dict.items():
            if road_id in important_roads:
                reward -= num_vehicles

        return reward

    def _take_action(self, action):
        if action != controlled_lights[0]['curr_phase']:
            controlled_lights[0]['curr_phase'] = action
            self._set_tl_phase(controlled_lights[0]['name'], action)

    @staticmethod
    def _get_road_waiting_vehicle_count():
        wait_counts = {'gneE16': 0, 'gneE59': 0, 'gneE13': 0}
        road_counts = {'gneE16': 0, 'gneE59': 0, 'gneE13': 0}
        vehicles = traci.vehicle.getIDList()
        for v in vehicles:
            road = traci.vehicle.getRoadID(v)
            if road in wait_counts.keys():
                if traci.vehicle.getWaitingTime(v) > 0:
                    wait_counts[road] += 1
                road_counts[road] += 1
        return wait_counts, road_counts

    def _on_training_end(self):
        super(self)
        traci.close()

    @staticmethod
    def _set_tl_phase(intersection_id, phase_id):
        traci.trafficlight.setPhase(intersection_id, phase_id)

    def render(self, mode='human', close=False):
        # self.save_replay = not self.save_replay
        self.current_binary = self.guiBinary
