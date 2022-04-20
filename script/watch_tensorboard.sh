#!/bin/bash
source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch-gpu
conda env list
latest_event=$(ls ~/coding_env/ML4PDR/log/ | grep tensorboard- | sort -nr | head -n 1)
tensorboard --logdir ~/coding_env/ML4PDR/log/$latest_event --port 8123