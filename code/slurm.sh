#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p gpu-share
#SBATCH -J myFirstGPUJob
#SBATCH --nodes=1                 # 申请一个节点
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1              # 每个节点上申请一块GPU卡

python train.py
