'''
Generate aag to extract graph
'''
from ast import arg
import subprocess
import sys
from natsort import natsorted
import argparse
from pathlib import Path
import os, os.path, shutil
from itertools import islice


def walkFile(dir):
    for root, _, files in os.walk(dir):
        files = natsorted(files)
        files = [os.path.join(root,f) for f in files if ".aag" not in f]
    return files

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def split_to_subset():
    sp = subprocess.Popen("du -b ../dataset/aag4train/* | sort -n", stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    return [line.decode("utf-8").strip('\n').split('\t') for line in sp.stdout.readlines()]

def remove_empty_file(list_of_file):
    for file in list_of_file:
        if file[0] == '0': subprocess.run(["trash-put", file[1]])
    return [file for file in list_of_file if file[0] != '0']
    
if __name__ == '__main__':
    
    '''
    --------------------main function--------------------
    '''
    aag_dir = str(Path(__file__).parent.parent)+'/dataset/aag4train/'
    help_info = "Usage: python generate_aag.py <aig-dir>"
    parser = argparse.ArgumentParser(description="Convert aig to aag automatically")
    parser.add_argument('-indir', type=str, default=None, help='Input the aiger directory name for aig to convert to aag')
    parser.add_argument('-outdir', type=str, default=aag_dir, help='Export the converted aag to the directory')
    parser.add_argument('-d', type=int, default=1, help='Determin whether to divide files into subset')
    parser.add_argument('-n', type=int, default=10, help='Determin how many files to divide into subset')
    args = parser.parse_args(['-indir','../dataset/aig_benchmark/hwmcc07_tip/','-outdir','../dataset/aag4train/','-n', '5'])
    #args = parser.parse_args()
    if args.indir is not None:
        aig_dir = args.indir
        file_lst = walkFile(aig_dir)

        for file in file_lst:
            # TODO: Use repr for this, make this command can be ran on Linux and Windows -> avoid Escape Character when use str
            # For instance, str(abc\ndef) will throw exception, using r'xxx' or repr() can avoid this problem
            cmd = [str(Path(__file__).parent/'aiger_tools/aigtoaig'), file, '-a']
            with open(args.outdir + file.split('/')[-1].replace('.aig','.aag'), "w") as outfile:
                subprocess.run(cmd, stdout=outfile)
    
    '''
    -------------------sort the file by size and split into chunks-------------------
    '''
    if args.d != 0:
        lst = split_to_subset()
        list_removed_empty = remove_empty_file(lst)
        list_chunks = list(chunk(list_removed_empty, args.n))
        for i_tuple in range(len(list_chunks)):
            if not os.path.isdir("../dataset/aag4train/subset_"+str(i_tuple)): 
                os.makedirs("../dataset/aag4train/subset_"+str(i_tuple))
            for i_file in range(len(list_chunks[i_tuple])): 
                shutil.copy(list_chunks[i_tuple][i_file][1], "../dataset/aag4train/subset_"+str(i_tuple))