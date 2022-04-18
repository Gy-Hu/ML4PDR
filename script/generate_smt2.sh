#source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch_try_install
source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch-gpu
#conda activate pytorch-gpu
conda env list
cd code
python main.py --mode 1 -t 600 -p ../dataset/aag4train/subset_$1 -c -d ig
