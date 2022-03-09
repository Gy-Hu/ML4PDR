'''
Only used when run on HPC
'''
from simple_slurm import Slurm
import os
import time
from pathlib import Path
prefix_folder = Path(__file__).parent.parent

slurm = Slurm(
    job_name='test_simple_slurm',
    partition='gpu-share', # Determine the partition
    nodes=1, #If the program cannot MPI, this should equal to one
    ntasks_per_node=6, # If the program cannot support multi-thread (such as openmp), this should be one
    gres='gpu:1', # Determine how many GPUs to apply
    qos='qos_gpu-share', # determine the price and quality of running job. Check by using : sacctmgr show ass user=`whoami`  format=user,part,qos
    output=f'{prefix_folder}/log/test_simple_slurm.out',
    # mail_type #begin or end
    # mail_user #user_name@ust.hk
    # cpu_per_task
    # nodelist
    # exclude #exclude the node of application
)

slurm.sbatch('python train.py' + Slurm.SLURM_ARRAY_TASK_ID)
time.sleep(20)
# after run this, remeber to use 'tail -f test_simple_slurm.out' to show the output (this task name can be change as well)

'''
# Set the maximum runtime, uncomment if you need it
##SBATCH -t 48:00:00 #Maximum runtime of 48 hours

# Use 2 nodes and 80 cores
#SBATCH -N 2 -n 80

# Setup runtime environment if necessary 
# or you can source ~/.bashrc or ~/.bash_profile

# Execute applications in parallel 
srun -n 1 myapp1 &    # Assign 1 core to run application "myapp1" 
srun -n 1 myapp2 &    # Similarly, assign 1 core to run application "myapp2" 
srun -n 1 myapp3 

# Setup runtime environment if necessary
# For example, setup intel MPI environment
module swap gnu8 intel

# Go to the job submission directory and run your application
cd $HOME/apps/slurm
mpirun ./your_mpi_application
'''