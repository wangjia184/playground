#!/bin/bash
IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch_cuda_dcae:latest
docker build --progress=plain -t=$IMAGE_URL .
docker push $IMAGE_URL

