#!/bin/bash

IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch_cuda_cifar10:latest
docker rm -f sde_diff
docker run -ti --name sde_diff --network=host --gpus all --runtime=nvidia \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/app \
    $IMAGE_URL $@


