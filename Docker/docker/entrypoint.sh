#!/bin/bash
# This script is the entrypoint for the Docker image.
# Taken from https://github.com/openai/gym/

#source /root/code/venv/bin/activate
#python -m pip uninstall -y stable-baselines
#python -m pip install stable-baselines[mpi]==2.10.0
#python -m pip uninstall -y mpi4py
#python -m pip install mpi4py
#sudo apt-get update update
#apt-get install -y sumo sumo-tools sumo-doc
add-apt-repository ppa:sumo/stable
apt-get update && apt-get install -y sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo

git clone https://github.com/Sahmwell/G15_Capstone.git  # Can probably remove if you're mounting it 
cd G15_Capstone/TrainingGym
python3.7 -m pip install --upgrade pip
python3.7 -m pip install -r venv/requirements.txt 
exec "$@"

