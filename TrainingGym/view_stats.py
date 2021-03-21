import numpy as np
import matplotlib.pyplot as plt
import json

# Load configs
with open('global_config.json') as global_json_file:
    global_config_params = json.load(global_json_file)
    local_config_path = global_config_params['config_path']
with open(f'Scenarios/{local_config_path}') as json_file:
    config_params = json.load(json_file)

controlled_lights = config_params['controlled_lights']
for i in range(len(controlled_lights) - 1, -1, -1):
    if not controlled_lights[i]['train']:
        del controlled_lights[i]


def main():
    data = np.load(input('Filename: '))
    for i_node in range(len(controlled_lights)):
        if i_node not in []:
            plt.plot(data[:, i_node], label=controlled_lights[i_node]['node_name'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
