## How to generate route files and Run Simulation:

**Generate the initial route file as follows**

```
python TrainingGym/tools/jtcrouter.py -n TrainingGym/Scenarios/VitalTri/VitalTriAM/VitalTriAM.net.xml -t TrainingGym/Scenarios/VitalTri/VitalTriAM/VitalTriAM.turn.xml -o TrainingGym/Scenarios/VitalTri/VitalTriAM/VitalTriAM.rou.xml --fringe-flows
```

**Optimize it using route sampler as follows:**

```
python TrainingGym/tools/routeSampler.py -r TrainingGym/Scenarios/VitalTri/VitalTriAM/VitalTriAM.rou.xml -t TrainingGym/Scenarios/VitalTri/VitalTriAM/VitalTriAM.turn.xml -o TrainingGym/Scenarios/VitalTri/VitalTriAM/VitalTriAM_Optimized.rou.xml --optimize full
```

**Load and run simulation in sumo from net edit as follows:**

1. Open NETEDIT
2. File -> Open Network
3. Select VitalTriAM.net.xml
4. Click OK
5. File -> Demand Elements -> Load Demand Elements
6. Select VitalTriAM_Optimized.rou.xml
7. Click OK
8. CTRL - T (or Edit -> Open in sumo-gui)
9. Select VitalTriAM_Optimized.rou.xml
10. Click OK
11. When prompted about files existing click Yes.
12. Click Run or press CTRL-A
13. Adjust Delay if you would like to observe traffic
