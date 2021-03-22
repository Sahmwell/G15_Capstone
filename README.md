Decentralized Intelligent Traffic Control:
Optimizing Vehicle Traffic Flow Using
Deep Reinforcement Learning in
Simulated Environments
---------------
The goal of our project was to create a Decentralized Intelligent Traffic Control (dITC) scheme that
optimized traffic flow in an area containing nine signalized intersections in the St. Vital neighbourhood in
Winnipeg during the morning rush hour. Our strategy for creating the dITC scheme for the intersections was to
use Reinforcement Learning (RL) in a simulated environment on a traffic model of our area of study. To create
this model, we used existing traffic data and
interfaced the traffic simulator with existing RL libraries to create a reward function that our RL
agent could train on. After a sufficient amount of time spent training our agent, our agents had
learned a traffic light schedule that when benchmarked resulted in improved traffic performance
metrics. In addition to our traffic performance metrics being realized, we were able to meet the
project performance metrics we set out to achieve in our second design review, as well as two extra
performance metrics we established during the implementation of our project. There were two
major resultsfrom our capstone project, the first being a portable, scalable, traffic-scenario-agnostic
system able to take advantage of high-performance computing resources that can be used to train
agents for the purpose of traffic optimization. The second result is learned policies for our RL
agents that when applied in our traffic simulator resulted in improved traffic performance metrics
during the morning rush hour through a dITC scheme.

File Structure
--------------
![g15_capstone_file_structure](https://user-images.githubusercontent.com/43391149/111924984-f1a69e00-8a74-11eb-9089-f7156f2a2bb8.png)
The above image illustrates the file structure of our project repository. The root directory contains three subfolders: 
Individual Code, Docker, and TrainingGym. Individual Code contains prototype code that does not belong to the core project code, while the Docker directory contains files related to building and running our Docker image. TrainingGym
is a Python project containing the codebase created for this project which contains Python
scripts used to train, test, and benchmark our agents, as well as the global configuration file. There
are also three directories within the TrainingGym folder: Scenarios, env, and venv. The env folder
contains code used to create the RL environment, while the venv folder contains the files used
to create the project’s virtual environment. The Scenarios directory contains all the traffic
scenarios the agents can train on, with configuration files for each scenario found in each of the
directories contained within.

Copyright © 2021, Samuel Anderson, Cody Au, Michael Guevarra Eric Nazer. Released under the MIT License.
