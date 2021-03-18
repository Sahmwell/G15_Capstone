FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN \
    apt-get update && apt-get install -y software-properties-common &&\
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y git sumo python3.7 python3.7-dev python3-pip cmake libopenmpi-dev ssh  zlib1g-dev

RUN \
    cd /root && \
    git clone https://github.com/Sahmwell/G15_Capstone.git

RUN \
    python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install -r /root/G15_Capstone/TrainingGym/venv/requirements.txt

RUN \
    add-apt-repository ppa:sumo/stable && apt-get update && apt-get install -y sumo sumo-tools sumo-doc

ENV SUMO_HOME=/usr/share/sumo

CMD /bin/bash
