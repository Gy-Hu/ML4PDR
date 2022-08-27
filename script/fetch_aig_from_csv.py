'''
Fetch aiger from Zhang's csv file, and convert it to aag
'''


# Fetch small aiger from /data/hongcezh/clause-learning/data-collect/stat/

import pandas as pd
from ast import arg
import subprocess
import sys
from matplotlib.cbook import index_of
from natsort import natsorted
import argparse
from pathlib import Path
import os, os.path, shutil
from itertools import islice

# appen file path to the system path
sys.path.append(f'{str(Path(__file__).parent.parent)}/code/')
from generate_aag import walkFile, chunk, remove_empty_file, remove_trivially_unsat_aiger

def fetch_aig_from_csv(csv_file):
    # Read this csv file into dataframe
    df = pd.read_csv(csv_file)

    # Then sort the dataframe by 'Res' and 'N_clauses'
    df = df[df["res"] == "unsat"].sort_values(['res','n_clause'], ascending = True).head(50)

    # Export the aag_name column to a list
    aag_list = df["aag_name"].tolist()

    # Add file path to the aag_list
    for i in range(len(aag_list)):
        aag_list[i] = "/data/hongcezh/clause-learning/data-collect/hwmcc20/" + aag_list[i] + ".aig"

    return aag_list

if __name__ == '__main__':
    aag_dir = f'{str(Path(__file__).parent.parent)}/dataset/aag4train_hwmcc20/'
    parser = argparse.ArgumentParser(description="Convert aig to aag automatically")
    parser.add_argument('-outdir', type=str, default=aag_dir, help='Export the converted aag to the directory')
    parser.add_argument('-d', type=int, default=1, help='Determin whether to divide files into subset')
    parser.add_argument('-n', type=int, default=10, help='Determin how many files to divide into subset')
    args = parser.parse_args(['-n', '5'])
    '''
    --------------------Get the aig list (and their path)-------------------
    '''
    csv_file = "/data/hongcezh/clause-learning/data-collect/stat/size20.csv"
    aag_list = fetch_aig_from_csv(csv_file)

    for file in aag_list:
        # TODO: Use repr for this, make this command can be ran on Linux and Windows -> avoid Escape Character when use str
        # For instance, str(abc\ndef) will throw exception, using r'xxx' or repr() can avoid this problem
        cmd = [str(Path(__file__).parent.parent/'code/aiger_tools/aigtoaig'), file, '-a']
        with open(args.outdir + file.split('/')[-1].replace('.aig','.aag'), "w") as outfile:
            subprocess.run(cmd, stdout=outfile)

    '''
    -------------------sort the file by size and split into chunks-------------------
    '''
    if args.d != 0:
        sp = subprocess.Popen("du -b ../dataset/aag4train_hwmcc20/* | sort -n", stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        lst = [line.decode("utf-8").strip('\n').split('\t') for line in sp.stdout.readlines()]
        list_removed_empty = remove_empty_file(lst)
        list_removed_trivial_unsat = remove_trivially_unsat_aiger(list_removed_empty)
        list_chunks = list(chunk(list_removed_trivial_unsat, args.n))
        for i_tuple in range(len(list_chunks)):
            if not os.path.isdir(f"../dataset/aag4train_hwmcc20/subset_{str(i_tuple)}"): 
                os.makedirs(f"../dataset/aag4train_hwmcc20/subset_{str(i_tuple)}")
            for i_file in range(len(list_chunks[i_tuple])): 
                shutil.copy(list_chunks[i_tuple][i_file][1], f"../dataset/aag4train_hwmcc20/subset_{str(i_tuple)}")

