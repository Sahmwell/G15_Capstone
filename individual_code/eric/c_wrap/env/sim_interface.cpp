#include <iostream>
#include <time.h>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <utils/traci/TraCIAPI.h>

using namespace std;

void connect(TraCIAPI client) {
    client.connect("localhost", 1337);
    cout << "1" << endl << flush;
}

// Go one step in simulation
void step(TraCIAPI client) {
    client.simulationStep();
    cout << "1" << endl << flush;
}

void stop(TraCIAPI client) {
    client.close();
}

void load(TraCIAPI client) {
    vector<string> arguments{"-c", "PoundSign/PoundSign.sumocfg", "-t", "--remote-port", "1337"};
    client.load(arguments);
}

// Set a traffic light phase

// Get waiting and total vehicle counts

//def _set_tl_phase(intersection_id, phase_id):
//    traci.trafficlight.setPhase(intersection_id, phase_id)
//
//def _get_road_waiting_vehicle_count():
//    wait_counts = {'gneE16': 0, 'gneE59': 0, 'gneE13': 0}
//    road_counts = {'gneE16': 0, 'gneE59': 0, 'gneE13': 0}
//    vehicles = traci.vehicle.getIDList()
//    for v in vehicles:
//        road = traci.vehicle.getRoadID(v)
//        if road in wait_counts.keys():
//            if traci.vehicle.getWaitingTime(v) > 0:
//                wait_counts[road] += 1
//            road_counts[road] += 1
//    return wait_counts, road_counts

int main(int argc , char** argv) {
    TraCIAPI client;
    bool done = false;
    string command;
    while(!done) {
        getline(cin, command);
        if (command == "connect"){
            connect(client);
        }
        else if(command == "load") {
            load(client);
        }
        else if(command == "step") {
            step(client);
        }
        else if(command == "stop") {
            stop(client);
            done = true;
        }
    }


}