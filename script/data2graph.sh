#source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch_try_install
source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch-gpu
#conda activate pytorch-gpu
conda env list
cd code
python data_gen.py -d ../dataset/IG2graph/generalize_IG/ -m ig
