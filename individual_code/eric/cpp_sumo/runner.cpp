#include <iostream>
#include <time.h>
#include <fstream>
#include <utils/traci/TraCIAPI.h>

using namespace std;

class Client : public TraCIAPI {
public:
    Client() {};
    ~Client() {};
};

void generate_routefile() {
    srand( (unsigned)time( NULL ) );
    int N = 24 * 3600;
    double probability = 1. / 10;
    ofstream routes;
    routes.open("data/PoundSign.rou.xml");
    routes << "<routes>" << endl;
    routes << "<vType id=\"default\" accel=\"0.8\" decel=\"4.5\" sigma=\"0.5\" length=\"5\" minGap=\"2.5\" maxSpeed=\"16.67\" guiShape=\"passenger\"/>" << endl;
    routes << "<route id=\"right\" edges=\"gneE65 gneE19 gneE55\" />" << endl;
    
    for (int i = 0; i < N; i++) {
        if ((double) rand()/RAND_MAX < probability) {
            routes << "    <vehicle id=\"test_" << to_string(i) << "\" type=\"default\" route=\"right\" depart=\"" << to_string(i) << "\" />" << endl;
        }
    }
    routes << "</routes>" << endl;
    routes.close();
    
}


void run(Client client) {
    int step = 0;
    int interval_count = 0;
    vector<string> last_vehicles;
    while (client.simulation.getMinExpectedNumber() > 0) {
        client.simulationStep();
        vector<string> vehicles = client.inductionloop.getLastStepVehicleIDs("nw_s");
        for (int i = 0; i < vehicles.size(); i++) {
            bool was_last = false;
            for (int j =0; j < last_vehicles.size() && !was_last; j++) {
                was_last = vehicles.at(i).compare(last_vehicles.at(j)) == 0;
            }
            if (!was_last)
                interval_count++;

        }
        last_vehicles = vehicles;

        if ((step % (15*60)) == 0) {
            cout << interval_count << endl;
            interval_count = 0;
        }
        step += 1;
    }
    client.close();
    cout << flush;
}

int main(int argc, char* argv[]) {
    generate_routefile();
    Client client;
    client.connect("localhost", 1337);
    run(client); 
    cout << "end of execution" << endl;
    // Use exit instead of return... for some reason when I do return 0 I get an error about something being free'd twice
    exit(EXIT_SUCCESS); 
}
