#!/bin/bash

IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch:2.6.0-cuda12.4-cudnn9-runtime
docker rm -f vqvae
docker run -ti --name vqvae --network=host --gpus all --runtime=nvidia \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/app \
    $IMAGE_URL $@


