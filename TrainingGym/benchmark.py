import traci
import json
import sumolib
import time
import random

from typing import List, Dict, Set
from stable_baselines import PPO2
from env.SumoEnvParallel import SumoEnvParallel
from collections import defaultdict

OUTPUT_TURN_COUNTS = False  # Set to True to Enable Turn Count Output
USE_LEARNED_MODEL = False  # Set to True to Benchmark PPO2 Model
OUTPUT_INTERSECTION_STATS = True  # Output Per Intersection Statistics
SEED = 500 # Set to -1 for random seed

def printProgressBar(iteration, total, prefix='Progress:', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    Source https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="")
    # Print New Line on Complete
    if iteration == total:
        print()


def get_metrics(
        connections: List[Dict], capacities: List[Dict], vehicles: Dict[str, float], all_important_roads: Dict,
        node_stats: Dict, t=traci, last_step_edge_cache=defaultdict(lambda: ""), wait_cache=defaultdict(int),
        total_wait_cache=defaultdict(int), routes={}
):
    for v_id in set(t.vehicle.getIDList()).difference(set(t.simulation.getDepartedIDList()).union(
            *t.simulation.getStopStartingVehiclesIDList()).union(*t.simulation.getStopEndingVehiclesIDList())):
        curr_wait = t.vehicle.getWaitingTime(v_id)
        edge = ""
        if OUTPUT_INTERSECTION_STATS:
            # Get Vehicle Edge
            if v_id not in routes:
                routes[v_id] = t.vehicle.getRoute(v_id)
            route_index = t.vehicle.getRouteIndex(v_id)
            edge = routes[v_id][route_index]
            if edge in all_important_roads:
                node_name = all_important_roads[edge]
                node_stats[node_name]['v_ids'].add(v_id)

        if not curr_wait and wait_cache[v_id]:
            total_wait_cache[v_id] += wait_cache[v_id]

            if OUTPUT_INTERSECTION_STATS:
                # Get Vehicle Edge
                last_edge = last_step_edge_cache[v_id]
                # If the last edge is on an important road add the wait to that nodes wait time
                if last_edge in all_important_roads:
                    node_name = all_important_roads[last_edge]
                    node_stats[node_name]['wait_sum'] += wait_cache[v_id]
        wait_cache[v_id] = curr_wait
        if OUTPUT_INTERSECTION_STATS:
            last_step_edge_cache[v_id] = edge
    for v_id in t.simulation.getArrivedIDList():
        if v_id not in vehicles:
            vehicles[v_id] = total_wait_cache[v_id]
        else:
            wait = total_wait_cache[v_id]
            print(f"Error overwriting vehicle ID {v_id} old val {vehicles[v_id]} new val {wait}")
            vehicles[v_id] = wait

    for direction in connections:
        [direction['vehicles'].add(v_id) for e in direction['edges'] for v_id in t.edge.getLastStepVehicleIDs(e)]
        direction['queues'].append(sum([t.edge.getLastStepHaltingNumber(e) for e in direction['edges']]))

    if OUTPUT_TURN_COUNTS:
        for capacity in capacities:
            for lane in capacity['unique_lanes']:
                [capacity['vehicles'].add(v_id) for v_id in t.lane.getLastStepVehicleIDs(lane)]


def step(
        connections: List[Dict], capacities: List[Dict], total_steps: int, vehicles: Dict[str, float], env, model, obs,
        all_important_roads, node_stats
):
    if USE_LEARNED_MODEL:
        action, state = model.predict(obs)
        new_obs, _, _, _ = env.step(action)
    else:
        new_obs = None
        traci.simulationStep()
    get_metrics(
        connections, capacities, vehicles, all_important_roads, node_stats, env.sumo if USE_LEARNED_MODEL else traci)

    t = traci.simulation.getTime()
    printProgressBar(t, total_steps, suffix=f'Complete. Finished {t} of {total_steps} ')
    return new_obs

def benchmark():
    t1 = time.time()
    # init
    with open('global_config.json') as global_json_file:
        global_config_params = json.load(global_json_file)
        local_config_path = global_config_params['config_path']
    with open(f'Scenarios/{local_config_path}') as json_file:
        config_params = json.load(json_file)

    seed = random.randint(0, 2**32 - 1) if SEED == -1 else SEED

    load_options = ["-c", f'Scenarios/{config_params["sumocfg_path"]}', "--start", "--quit-on-end",
                    "--seed", str(seed), "--no-warnings", "true"]

    # Get config parameters
    total_steps = config_params["test_steps"]
    controlled_lights = config_params['controlled_lights']
    connections = [direction for intersection in controlled_lights for direction in intersection['connections']]
    capacities = [capacity for intersection in controlled_lights for capacity in intersection['capacities']]
    vehicles = {}
    for capacity in capacities:
        capacity['vehicles'] = set()
    for direction in connections:
        direction['vehicles'], direction['queues'], direction['delays'] = set(), [], []
    all_important_roads = {e: n['node_name'] for n in controlled_lights for d in n['connections'] for e in d['edges']}
    node_stats = {}
    for node in controlled_lights:
        node_stats[node['node_name']] = {
            'edges': {edge for direction in node['connections'] for edge in direction},
            'wait_sum': 0,
            'v_ids': set()
        }

    env, model, obs = None, None, None
    if USE_LEARNED_MODEL:
        controlled_lights = config_params['controlled_lights']
        for i in range(len(controlled_lights) - 1, -1, -1):
            if not controlled_lights[i]['train']:
                del controlled_lights[i]
        env = SumoEnvParallel(config_params['test_steps'], True, controlled_lights[0]['light_name'],
                              collect_statistics=False, seed=SEED)
        model = PPO2.load(f'Scenarios/{config_params["model_save_path"]}/PPO2_{controlled_lights[0]["light_name"]}')
        obs = env.reset()

    if not USE_LEARNED_MODEL:
        traci.start([sumolib.checkBinary('sumo')] + load_options)
    t = traci if not USE_LEARNED_MODEL else env.sumo
    for i in range(total_steps):
        obs = step(connections, capacities, total_steps, vehicles, env, model, obs, all_important_roads, node_stats)
        if not t.simulation.getMinExpectedNumber():
            print(f"Simulation ended at {t.simulation.getTime()}.")
            break
    avg_queue_length = sum([q_len for direction in connections for q_len in direction['queues']]) / sum(
        [len(direction['queues']) for direction in connections])
    avg_delay = sum([wait_time for wait_time in vehicles.values()]) / len(vehicles)

    t.close()

    def condition(arrive_rate, road_cap):
        norm_arrive_rate = 3600 * (arrive_rate / total_steps)
        return (norm_arrive_rate - road_cap) / max(road_cap, 1) if norm_arrive_rate > road_cap else 0

    edges = [edge for edge in connections if edge['capacity']]
    avg_overflow = sum([condition(len(edge['vehicles']), edge['capacity']) for edge in edges]) / len(edges)

    print(f'Average Overflow (%): {avg_overflow * 100}\n'
          f'Average Queue Length (Num Cars): {avg_queue_length}\n'
          f'Average Delay: {avg_delay}')
    print(f"Benchmark finished in {time.time() - t1} seconds")

    if OUTPUT_TURN_COUNTS:
        [print(f'Total Vehicles on {r["label"]}: {len(r["vehicles"])} Expected: {r["expected"]}') for r in connections]
        [print(f'Total Vehicles on {l["label"]}: {len(l["vehicles"])} Expected: {l["expected"]}') for l in capacities]

    if OUTPUT_INTERSECTION_STATS:
        all_node_directions = {key: [] for key in node_stats.keys()}
        for direction in edges:
            node_name = f"node_{direction['label'][0]}"
            all_node_directions[node_name].append(direction)
        for node, node_stat in node_stats.items():
            directions = all_node_directions[node]
            avg_node_overflow = sum(
                [condition(len(direction['vehicles']), direction['capacity']) for direction in directions]
            ) / (len(directions) if directions else 1)
            avg_node_queue_length = sum([q_len for direction in directions for q_len in direction['queues']]) / (
                sum([len(direction['queues']) for direction in directions]) if directions else 1)
            avg_node_delay = node_stat['wait_sum'] / (len(node_stat['v_ids']) if len(node_stat['v_ids']) else 1)

            print(f'{node}\n{"-"*10}\nAverage Overflow (%): {avg_node_overflow * 100}\n'
                  f'Average Queue Length (Num Cars): {avg_node_queue_length}\n'
                  f'Average Delay: {avg_node_delay}\n{"-"*20}')


benchmark()
