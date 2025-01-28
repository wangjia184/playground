#!/bin/bash
IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch:2.5.1-cuda12.4-cudnn9
docker build --progress=plain -t=$IMAGE_URL .
#docker push $IMAGE_URL

