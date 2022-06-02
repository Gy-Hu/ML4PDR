
#!/bin/bash

source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch-gpu
conda env list
python data_gen_no_enumerate.py -d ../dataset/IG2graph/generalize_IG_no_enumerate/ -m ig -s $1
