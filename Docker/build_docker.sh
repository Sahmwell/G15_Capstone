#!/bin/bash

#PARENT=stablebaselines/stable-baselines
PARENT=bionic

VERSION=v1.0
TAG=g15capstone/containerized-gpu

USE_GPU=True

if [[ ${USE_GPU} == "True" ]]; then
  PARENT="${PARENT}:${VERSION}"
else
  PARENT="${PARENT}-cpu:${VERSION}"
  TAG="${TAG}-cpu"
fi

# --build-arg PARENT_IMAGE=${PARENT} <- Removed since I think we don't want
docker build --build-arg USE_GPU=${USE_GPU} -t ${TAG}:${VERSION} . -f docker/Dockerfile
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
  docker push ${TAG}:${VERSION}
  docker push ${TAG}:latest
fi
