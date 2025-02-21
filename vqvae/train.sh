#!/bin/bash

echo "docker exec vqvae tensorboard --logdir /app/runs"
./run.sh python /app/train.py
