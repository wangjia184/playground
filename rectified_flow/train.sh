#!/bin/bash

echo "docker exec sde_diff tensorboard --bind_all --logdir /app/runs"
./run.sh python /app/train.py
