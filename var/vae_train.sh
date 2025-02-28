#!/bin/bash

echo "docker exec var tensorboard --logdir /app/runs"
./run.sh python /app/vqvae/train.py
