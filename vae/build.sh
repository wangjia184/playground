#!/bin/bash
IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch_cuda_carton:latest
docker build -t=$IMAGE_URL .
docker push $IMAGE_URL

