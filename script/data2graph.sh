
# Run in the code directory of the project

# has run
# cmu.gigamax.B
# eijk.S208.S
# eijk.S208c.S
# eijk.S208o.S
# eijk.S386.S
# eijk.S641.S
# eijk.S713.S
# ken.flash^01.C
# ken.oop^1.C
# ken.oop^2.C
# nusmv.syncarb10^2.B
# texas.ifetch1^5.Ed
# texas.ifetch1^8.E
# vis.4-arbit^1.E
# vis.arbiter.E
# vis.elevator^2.E
# vis.elevator^3.E


source /data/guangyuh/miniconda3/bin/activate /data/guangyuh/miniconda3/envs/pytorch-gpu
conda env list
python data_gen.py -d ../dataset/IG2graph/generalize_IG/ -m ig -s $1
