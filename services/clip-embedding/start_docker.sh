DOCKER_IMG=clip:latest
docker run --rm --name clip --gpus device=0 --shm-size 16G -p 20541:20541 -it -v $(pwd)/:/home/nhtlong/workspace/ ${DOCKER_IMG} /bin/bash
# docker run --rm --name clip --gpus device=0 --shm-size 16G -it ${DOCKER_IMG} /bin/bash