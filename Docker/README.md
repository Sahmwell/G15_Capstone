# How to build, run, and start training agents in a container
1. Open a bash shell
2. Run build_docker.sh
3. Run run_docker.sh
4. Open another bash shell and run the following command to get the name of the container. There should be a container 
running with the image name g15capstone/containerized-gpu. Take note of the corresponding NAME. 
    ```shell script
    docker ps
    ```
5. In the bash shell within the container adjust run the following command to start training an agent. 
    ```shell script
    python3.7 train.py
    ```

### Related workflow commands
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