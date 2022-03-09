#!/bin/bash

# Parameters
#SBATCH --error=/home/ghuae/coding_env/ML4PDR/code/../log/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/ghuae/coding_env/ML4PDR/code/../log/%j_0_log.out
#SBATCH --partition=gpu-share
#SBATCH --signal=USR1@90
#SBATCH --time=1
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /home/ghuae/coding_env/ML4PDR/code/../log/%j_%t_log.out --error /home/ghuae/coding_env/ML4PDR/code/../log/%j_%t_log.err --unbuffered /home/ghuae/miniconda3/envs/pytorch-gpu/bin/python -u -m submitit.core._submit /home/ghuae/coding_env/ML4PDR/code/../log