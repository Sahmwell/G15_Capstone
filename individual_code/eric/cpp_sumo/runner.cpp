#include <iostream>
#include <time.h>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <utils/traci/TraCIAPI.h>

using namespace std;

void generate_routefile() {
    srand( (unsigned)time( NULL ) );
    int N = 20 * 1000;
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


void run(TraCIAPI client) {
    int step = 0;
    int interval_count = 0;
    vector<string> last_vehicles;
    while (client.simulation.getMinExpectedNumber() > 0) {
        client.simulationStep();
        step += 1;
    }
    client.close();
    cout << flush;
}

int main(int argc, char* argv[]) {
    generate_routefile();
    TraCIAPI client;

    // Wait to make sure that the SUMO session is open since the library has no way to repeatedly attempt to connect
    sleep(2);

    client.connect("localhost", 1337);
    std::pair<int, std::string> versions = client.getVersion();
    cout << versions.first << endl;
    cout << versions.second << endl;

    run(client);
    cout << "end of execution" << endl;
    // Use exit instead of return... for some reason when I do return 0 I get an error about something being free'd twice
    exit(EXIT_SUCCESS); 
}
