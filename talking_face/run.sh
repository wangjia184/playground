#!/bin/bash

IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch:2.5.1-cuda12.4-cudnn9-talking-face
docker rm -f talking_face
docker run -ti --name talking_face --network=host --gpus all --runtime=nvidia \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/app \
    $IMAGE_URL $@


