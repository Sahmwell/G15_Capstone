#ARG PARENT_IMAGE
ARG USE_GPU
#FROM $PARENT_IMAGE
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

#RUN apt-get -y update \
#    && apt-get -y install \
#    ffmpeg \
#    freeglut3-dev \
#    swig \
#    xvfb \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*

#ENV CODE_DIR /root/code
#ENV VENV /root/code/venv

RUN \
    apt-get update && apt-get install -y software-properties-common &&\
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y git sumo python3.7 python3.7-dev python3-pip cmake libopenmpi-dev ssh  zlib1g-dev

#ENV PATH=$VENV/bin:$PATH

COPY docker/entrypoint.sh /tmp/
ENTRYPOINT ["/tmp/entrypoint.sh"]

CMD /bin/bash
