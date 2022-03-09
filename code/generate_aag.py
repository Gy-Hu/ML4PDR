'''
Generate aag to extract graph
'''
from ast import arg
import subprocess
import os
import sys
from natsort import natsorted
import argparse
from pathlib import Path


def walkFile(dir):
    for root, _, files in os.walk(dir):
        files = natsorted(files)
        files = [os.path.join(root,f) for f in files if ".aag" not in f]
    return files

if __name__ == '__main__':
    aag_dir = str(Path(__file__).parent/'dataset/aag4train/')
    help_info = "Usage: python generate_aag.py <aig-dir>"
    parser = argparse.ArgumentParser(description="Convert aig to aag automatically")
    parser.add_argument('-dir', type=str, default=None, help='Input the aiger directory name for aig to convert to aag')
    args = parser.parse_args()
    if args.dir is not None:
        aig_dir = args.dir
        file_lst = walkFile(aig_dir)

        for file in file_lst:
            cmd = [repr(str(Path(__file__).parent/'code/aiger_tools/aigtoaig')), file, '-a']
            with open(aag_dir + file.split('/')[-1].replace('.aig','.aag'), "w") as outfile:
                subprocess.run(cmd, stdout=outfile)