import gym
from gym import spaces, logger
import cityflow
import numpy as np

# HARDCODE
lights = [{'name':'gneJ8', 'curr_phase':0, 'num_phases': 10}]
important_lanes = ['gneE3_0', 'gneE3_1', 'gneE3_2', 'gneE4_0', 'gneE4_1', 'gneE4_2']

class LineEnv(gym.Env):
    def __init__(self, scenario_name, steps_per_episode):
        super(LineEnv, self).__init__()
        self.scenario_name = scenario_name
        self.eng = cityflow.Engine("data/" + self.scenario_name + "/config.json", thread_num=12)
        self.eng.set_save_replay(True)
        self.steps_per_episode = steps_per_episode
        self.is_done = False

        self.reward_range = (-float('inf'), float('inf')) # HARDCODE
        self.action_space = spaces.Discrete(10) # HARDCODE
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=np.array([12]), dtype=np.float32) # HARDCODE
    
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
        if action != lights[0]['curr_phase']:
            lights[0]['curr_phase'] = action
            self.eng.set_tl_phase(lights[0]['name'], action)

    def render(self, mode='human', close=False):
        # self.save_replay = not self.save_replay
        self.eng.set_save_replay(True)