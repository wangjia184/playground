#!/bin/bash

echo "docker exec var tensorboard --logdir /app/runs"
./run.sh python /app/vae/train.py
