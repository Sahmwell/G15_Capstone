#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse



# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary 
import traci 


# Number of simulation timesteps
N = 24 * 3600 

def main():
    # options = get_options()

    # if options.nogui:
    sumoBinary = checkBinary('sumo')
    # else:
    #     sumoBinary = checkBinary('sumo-gui')

    # # first, generate the route file for this simulation
    # generate_routefile(N)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "PoundSign/PoundSign.sumocfg",
                             "--tripinfo-output", "tripinfo.xml", "-t"])
    run()

def generate_routefile(N):

    with open("PoundSign/PoundSign.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="default" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <route id="sw-ne" edges="gneE65 gneE19 gneE16 gneE57" />
        <route id="ne-sw" edges="gneE59 gneE13 gneE12 gneE52" />""", file=routes)
        vehicle_id = 0
        for i in range(N):
            if i % 8 == 0:
                print(f'    <vehicle id="test_{vehicle_id}" type="default" route="sw-ne" depart="{i}" />', file=routes)
                vehicle_id += 1
                print(f'    <vehicle id="test_{vehicle_id}" type="default" route="ne-sw" depart="{i}" />', file=routes)
                vehicle_id += 1
        print("</routes>", file=routes)

def run():
    """execute the TraCI control loop"""
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step < 20000:
        traci.simulationStep()
        if step % 500 == 0:
            print(get_lane_waiting_vehicle_count())
            traci.load(["-c", "PoundSign/PoundSign.sumocfg",
                             "--tripinfo-output", "tripinfo.xml", "-t"])
        step += 1
    traci.close()
    sys.stdout.flush()

def get_lane_waiting_vehicle_count():
    counts = {'gneE16':0, 'gneE59':0, 'gneE13':0}
    vehicles = traci.vehicle.getIDList()
    for v in vehicles:
        road = traci.vehicle.getRoadID(v)
        if road in counts.keys() and traci.vehicle.getWaitingTime(v) > 0:
            counts[road] += 1
    return counts

def set_tl_phase(intersection_id, phase_id):
    traci.trafficlight.setPhase(intersection_id, phase_id)

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, _ = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    main()
