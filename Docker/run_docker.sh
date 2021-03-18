#!/bin/bash

cmd_line="$@"

echo "Executing in the docker:"
echo $cmd_line

TAG=g15capstone/containerized-gpu
VERSION=v1.0

docker run -it --runtime=nvidia --rm --network host --ipc=host -w /root/code ${TAG}:${VERSION}
#  --mount src=$(pwd)/../TrainingGym,target=/root/code,type=bind ${TAG}:${VERSION}

