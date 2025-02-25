#!/bin/bash
IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch:2.6.0-cuda12.4-cudnn9-runtime-face128x128
docker build -t=$IMAGE_URL .
docker push $IMAGE_URL

