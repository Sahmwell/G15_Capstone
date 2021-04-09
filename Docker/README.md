## How to build, run, and start training agents in a container
1. Open a bash shell
2. Run build_docker.sh
3. Run run_docker.sh
4. In the bash shell within the container run the following command to start training an agent. 
    ```shell script
    python3.7 train.py
    ```
    
## How to pull, run and start trianing agents in a container with Docker Hub
1. Pull the repository in either windows or linux: 
    ```shell script
    docker pull sahmwell/g15_capstone:latest
    ```
2. If you are on windows or linux without CUDA support run the following command: (Note that there is no CUDA support for running docker on windows)
    ```shell script
    docker run -it --rm --network host --ipc=host -w /root/G15_Capstone/TrainingGym sahmwell/g15_capstone
    ```
    If you are on linux with CUDA support run the following command (For CUDA support in docker see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    ```shell script
    docker run -it --runtime=nvidia --rm --network host --ipc=host -w /root/G15_Capstone/TrainingGym g15_capstone
    ```
3. In the bash shell within the container run the following command to start training an agent. 
    ```shell script
    python3.7 train.py
    ```

## Related workflow commands
Get the CONTAINER_NAME of the container in a bash shell on the host machine run the following command. There should be a 
container with the image name "*g15capstone/containerized-gpu*" running. Take note of the corresponding NAME. 

```shell script
docker ps
```

To copy files/folders to or from the container use the following commands respectively in a bash shell outside the 
container. This command can be used to copy Scenario files to the docker container and copy RL models and statistics out of the 
container.

```shell script
docker cp [SRC_PATH_ON_HOST] [CONTAINER_NAME]:[PATH_IN_CONTAINER]

docker cp [CONTAINER_NAME]:[SRC_PATH_IN_CONTAINER] [PATH_ON_HOST]
```

Additional bash shells can be opened within the container with the following command:

```shell script
docker exec -it [CONTAINER_NAME] /bin/bash
```

Clean up old containers and images (note this will remove any unused containers and images from any project):

```shell script
docker system prune
```

The docker image can be pulled from docker hub at the following URL

```
https://hub.docker.com/r/sahmwell/g15_capstone/tags?page=1&ordering=last_updated
```

The docker image can be pulled with the following command

```
docker pull sahmwell/g15_capstone:latest
```