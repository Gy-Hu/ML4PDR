
'''
Run this in code/ directory
'''
# get file list of IG2graph

import os
import argparse
from natsort import natsorted
import subprocess
import multiprocessing
import subprocess
import shlex
from multiprocessing.pool import ThreadPool

if __name__ == '__main__':
    subset_dir = '/data/guangyuh/coding_env/ML4PDR/dataset/aag4train/subset_'
    subset_dir_lst = [subset_dir+str(i) for i in range(1,22)] # non-trival
    
    pool = ThreadPool(multiprocessing.cpu_count())
    results = []
    for subset in subset_dir_lst[:]:
        cmd = "python main.py --mode 1 -t 3600 -p " + subset + " -c -r on -n on -a on -th 0.7 -mn neuropdr_2022-06-09_12:27:41_last -inf_dev gpu"
        subprocess.Popen(shlex.split(cmd)).wait()

    # Close the pool and wait for each running task to complete
    print("Finish all the subprocess, all the subset has been tested.")