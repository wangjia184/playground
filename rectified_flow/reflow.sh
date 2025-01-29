#!/bin/bash

echo "docker exec sde_diff tensorboard --logdir /app/runs"
./run.sh python /app/reflow.py
