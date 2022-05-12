#!/bin/bash
# Run in the code directory of the project
source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch-gpu
conda env list
python main.py --mode 1 -t 600 -p ../dataset/aag4train/subset_$1 -c -d ig
