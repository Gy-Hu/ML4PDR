
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

def call_proc(cmd):
    """ This runs in a separate thread. """
    #subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = p.communicate()
    return (_, err)


def walkFile(dir):
    for root, _, files in os.walk(dir):
        files = natsorted(files)
        files = [os.path.join(root,f) for f in files if ".aag" not in f]
    return files

if __name__ == '__main__':
    csv_dir = '/data/guangyuh/coding_env/ML4PDR/dataset/IG2graph/generalization_no_enumerate/'
    files = walkFile(csv_dir)
    files = [(f.split('/')[-1]).replace('.csv','') for f in files]
    
    pool = ThreadPool(multiprocessing.cpu_count())
    results = []
    for file in files:
        #args = shlex.split("./data2graph_no_enumerate.sh" + " " +file)
        results.append(pool.apply_async(call_proc, ("python data_gen_no_enumerate.py -d ../dataset/IG2graph/generalize_IG_no_enumerate/ -m ig -s " + file,)))

    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
    for result in results:
        _, err = result.get()
        print("err: {}".format(err))
    print("Finish all the subprocess, graph generation is ended.")
