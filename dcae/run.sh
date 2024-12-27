#!/bin/bash

IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch_cuda_dcae:latest
docker rm -f dcae
docker run -ti --name dcae --network=host --gpus all --runtime=nvidia \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/app \
    $IMAGE_URL  python /app/test.py

