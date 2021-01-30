## This file is a place to keep track of different reward and observation functions, and how they performed.
_Please give each function a somewhat descriptive name and add any relevant information / results._  

Since we'll be updating the environment code frequently we might nuke each other's reward or observation functions. So
to alleviate that issue, if you have a reward function or observation function you want to keep around when merging 
changes, or you have one you want to keep track of throw it in this file.  

----------------------------------------------------------------------------------------------------------------------  

#### -5 light switch with total wait time
Did great in the plus sign environment, but with multiple agents seemed to take a while to reach optimal solution
on multi-agent.
```python
def _get_reward(self):
    road_waiting_vehicles_dict, _, _ = self._get_direction_vehicle_counts(self.controlled_node)
    reward = 0.0
    if self.current_action != self.previous_action:
        reward -= 5

    for direction in self.controlled_node['connections']:
            reward -= road_waiting_vehicles_dict[direction['label']]

    return reward
```

#### -5 light switch with weighted wait time on switch
Potentially converged faster with multi-agent than above, but it is not well tested yet.
```python
def _get_reward(self):
    road_waiting_vehicles_dict, _, _ = self._get_direction_vehicle_counts(self.controlled_node)
    reward = 0.0
    if self.current_action != self.previous_action:
        reward -= 5

    for direction in self.controlled_node['connections']:
        if self.current_action != self.previous_action:
            reward -= 5 * road_waiting_vehicles_dict[direction['label']]
        else:
            reward -= road_waiting_vehicles_dict[direction['label']]

    return reward
```