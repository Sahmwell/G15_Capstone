#!/bin/bash

VERSION=v1.0
TAG=g15capstone/containerized-gpu

docker build -t ${TAG}:${VERSION} . -f Dockerfile
docker tag ${TAG}:${VERSION} ${TAG}:latest
