#!/bin/bash

echo "docker exec -t vae_dev tensorboard --logdir /app/runs"
./run.sh python /app/train.py
