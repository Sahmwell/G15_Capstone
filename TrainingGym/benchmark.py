import traci
import json
import sumolib


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


def get_metrics(connections: list):
    for direction in connections:
        [direction['vehicles'].add(v_id) for e in direction['edges'] for v_id in traci.edge.getLastStepVehicleIDs(e)]
        direction['delays'].append(sum([traci.edge.getWaitingTime(e) for e in direction['edges']]))
        direction['queues'].append(sum([traci.edge.getLastStepHaltingNumber(e) for e in direction['edges']]))


def step(connections: list, total_steps: int):
    # Advance simulation
    traci.simulationStep()
    get_metrics(connections)

    t = traci.simulation.getTime()
    printProgressBar(t, total_steps, suffix=f'Complete. Finished {t} of {total_steps} ')


def benchmark():
    # init
    with open('global_config.json') as global_json_file:
        global_config_params = json.load(global_json_file)
        local_config_path = global_config_params['config_path']
    with open(f'Scenarios/{local_config_path}') as json_file:
        config_params = json.load(json_file)

    load_options = ["-c", f'Scenarios/{config_params["sumocfg_path"]}', "--tripinfo-output",
                    f'Scenarios/{config_params["tripinfo_output_path"]}', "-t", "--random"]

    # Get config parameters
    total_steps = config_params["steps_per_episode"]
    controlled_lights = config_params['controlled_lights']
    connections = [direction for intersection in controlled_lights for direction in intersection['connections']]
    for direction in connections:
        direction['vehicles'], direction['queues'], direction['delays'] = set(), [], []

    traci.start([sumolib.checkBinary('sumo')] + load_options)
    [step(connections, total_steps) for _ in range(total_steps)]

    avg_queue_length = sum([q_len for direction in connections for q_len in direction['queues']]) / sum(
        [len(direction['queues']) for direction in connections])
    avg_delay_on_important_edges = sum([delay for direction in connections for delay in direction['delays']]) / sum(
        [len(direction['delays']) for direction in connections])
    count = traci.vehicle.getIDCount() if traci.vehicle.getIDCount() else 1
    avg_v_delay = sum([traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()]) / count

    traci.close()

    def condition(arrive_rate, road_cap):
        norm_arrive_rate = 3600 * (arrive_rate / total_steps)
        return (norm_arrive_rate - road_cap) / road_cap if norm_arrive_rate > road_cap else 0

    avg_overflow = sum([condition(len(edge['vehicles']), edge['capacity']) for edge in connections]) / len(connections)

    print(f'Average Overflow (%): {avg_overflow}\n'
          f'Average Queue Length (Num Cars): {avg_queue_length}\n'
          f'Average Delay On Edge (seconds): {avg_delay_on_important_edges}\n'
          f'Average Vehicle Wait (seconds): {avg_v_delay}')

    [print(f'Total Vehicles on {r["label"]}: {len(r["vehicles"])} Expected: {r["expected"]}') for r in connections]


benchmark()
