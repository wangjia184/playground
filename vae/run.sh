#!/bin/bash

IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch_cuda:latest
docker rm -f vae_dev
docker run -ti --name vae_dev --network=host --gpus all --runtime=nvidia \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/app \
    $IMAGE_URL $@


