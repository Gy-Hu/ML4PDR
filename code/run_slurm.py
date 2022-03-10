'''
Only used when run on HPC
'''
import argparse
from simple_slurm import Slurm
import os
import time
import subprocess
import sys
from pathlib import Path
prefix_folder = Path(__file__).parent.parent


if __name__ == '__main__':
    help_info = "Usage: python run_slurm.py <mode>"
    parser = argparse.ArgumentParser(description="Submit the job to slurm or check the log file")
    parser.add_argument('--mode',type=int,help='choose the mode to run the program, 0 means submit job, 1 means check the output log',default=0)
    parser.add_argument('--job-name',type=str,help='give the job name of this task')
    # test usage
    #args = parser.parse_args(['--job-name','test'])
    #args = parser.parse_args(['--mode',1,'--job-name','test'])
    
    args = parser.parse_args()

    if args.job_name is not None:
        job_log = args.job_name+".out"

    else:
        print("Wrong input, please check.")
    
    slurm = Slurm(
        job_name=args.job_name,
        partition='gpu-share', # Determine the partition
        nodes=1, #If the program cannot MPI, this should equal to one
        ntasks_per_node=6, # If the program cannot support multi-thread (such as openmp), this should be one
        gres='gpu:1', # Determine how many GPUs to apply
        qos='qos_gpu-share', # determine the price and quality of running job. Check by using : sacctmgr show ass user=`whoami`  format=user,part,qos
        output=f'{prefix_folder}/log/{job_log}',
        # mail_type #begin or end
        # mail_user #user_name@ust.hk
        # cpu_per_task
        # nodelist
        # exclude #exclude the node of application
    )

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

    if args.mode==0:
        slurm.sbatch('python train.py' + Slurm.SLURM_ARRAY_TASK_ID)
        # after run this, remeber to use 'tail -f test_simple_slurm.out' to show the output (this task name can be change as well)

    elif args.mode == 1:
        cmd = ['tail', '-f', f'{prefix_folder}/log/{job_log}']
        subprocess.run(cmd)

    else:
        print("Wrong input, check the help document again")

    