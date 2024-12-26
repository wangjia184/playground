#!/bin/bash

echo "docker exec vae_dev tensorboard --logdir /app/runs"
./run.sh python /app/train.py
