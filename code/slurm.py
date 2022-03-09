from simple_slurm import Slurm
import os
import time

slurm = Slurm(
    job_name='test_simple_slurm',
    partition='gpu-share',
    nodes=1,
    ntasks_per_node=6,
    gres='gpu:1',
    qos='qos_gpu-share',
    output=f'test_simple_slurm.out',
)
#slurm.sbatch('python train.py' + Slurm.SLURM_ARRAY_TASK_ID)
time.sleep(20)
os.system("tail -f test_simple_slurm.out")