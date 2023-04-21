#!/bin/bash
source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch-gpu
conda env list
latest_event=$(ls ~/coding_env/ML4PDR/log/ | grep tensorboard- | sort -nr | head -n 1)
tensorboard --logdir ~/coding_env/ML4PDR/log/$latest_event --port 8123
#tensorboard --logdir ~/coding_env/ML4PDR/log/$latest_event --host=192.168.8.44 --port 8123