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
    aag_dir = str(Path(__file__).parent.parent)+'/dataset/aag4train/'
    help_info = "Usage: python generate_aag.py <aig-dir>"
    parser = argparse.ArgumentParser(description="Convert aig to aag automatically")
    parser.add_argument('-indir', type=str, default=None, help='Input the aiger directory name for aig to convert to aag')
    parser.add_argument('-outdir', type=str, default=aag_dir, help='Export the converted aag to the directory')
    #args = parser.parse_args(['-indir','../dataset/aig_benchmark/beem/','-outdir','../dataset/aig_benchmark/beem_aag/'])
    args = parser.parse_args()
    if args.indir is not None:
        aig_dir = args.indir
        file_lst = walkFile(aig_dir)

        for file in file_lst:
            '''
            TODO: Use repr for this, make this command can be ran on Linux and Windows -> avoid Escape Character when use str
            For instance, str(abc\ndef) will throw exception, using r'xxx' or repr() can avoid this problem
            '''
            cmd = [str(Path(__file__).parent/'aiger_tools/aigtoaig'), file, '-a']
            with open(args.outdir + file.split('/')[-1].replace('.aig','.aag'), "w") as outfile:
                subprocess.run(cmd, stdout=outfile)