import gym
from gym import spaces, logger
import cityflow
import numpy as np

# HARDCODE
controlled_lights = [{'name':'ne', 'curr_phase':0, 'num_phases': 5}]
uncontrolled_lights = [{'name':'nw', 'curr_phase':0, 'num_phases': 4}, {'name':'se', 'curr_phase':0, 'num_phases': 4}, {'name':'sw', 'curr_phase':0, 'num_phases': 4}]
important_lanes = ['gneE16_0', 'gneE59_0', 'gneE13_0']

class CityEnv(gym.Env):
    def __init__(self, scenario_name, steps_per_episode):
        super(CityEnv, self).__init__()
        self.scenario_name = scenario_name
        self.eng = cityflow.Engine("data/" + self.scenario_name + "/config.json", thread_num=12)
        self.eng.set_save_replay(True)
        self.steps_per_episode = steps_per_episode
        self.is_done = False

        self.reward_range = (-float('inf'), float('inf')) # HARDCODE
        self.action_space = spaces.Discrete(5) # HARDCODE
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=np.array([6]), dtype=np.float32) # HARDCODE
    
    def reset(self):
        # self.eng.set_tl_phase('nw', 2) # HARCODE
        self.eng.reset()
        self.current_step = 0
        self.is_done = False

        return self._next_observation()

    def _next_observation(self):
        obs = []
        lane_counts = self.eng.get_lane_vehicle_count()
        wait_counts = self.eng.get_lane_waiting_vehicle_count()

        # HARDCODE
        for lane in important_lanes:
            obs.append(lane_counts[lane])
            obs.append(wait_counts[lane])

        return np.array(obs)
    
    def step(self, action):

        self._take_action(action)
        if self.current_step % 50 == 0:
            for light in uncontrolled_lights:
                light['curr_phase'] = (light['curr_phase'] + 1) % light['num_phases']
                self.eng.set_tl_phase(light['name'], light['curr_phase'])

        self.eng.next_step()
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
        lane_waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        reward = 0.0

        for (lane_id, num_vehicles) in lane_waiting_vehicles_dict.items():
            if lane_id in important_lanes:
                reward -= num_vehicles

        return reward

    def _take_action(self, action):
        if action != controlled_lights[0]['curr_phase']:
            controlled_lights[0]['curr_phase'] = action
            self.eng.set_tl_phase(controlled_lights[0]['name'], action)

    def render(self, mode='human', close=False):
        # self.save_replay = not self.save_replay
        self.eng.set_save_replay(True)