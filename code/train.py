import argparse
import pickle
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from config import parser
from neuro_predessor import NeuroPredessor
from data_gen import problem

#TODO: setup the batch size, iteration and epoch
'''
dataset -> contains around 10000 cases
batch -> one .smt2? (one time of generalized predecessor?) 
iteration -> every .smt2
epoch -> len(dataset)/batch -> 10000/1 maybe?
'''

if __name__ == "__main__":

  args = parser.parse_args(['--task-name', 'neuropdr_no1', '--dim', '128', '--n_rounds', '26', \
                            '--epochs', '5', \
                            '--gen_log', '/home/gary/coding_env/NeuroSAT/log/data_maker_sr3t10.log', \
                            '--train-file', '../dataset/train/eijk.S208o.S_0.pkl'
                            ])


  #TODO: refine the ground truth (which is the is_flexible[] list) with MUST here
  #data = "../dataset/train/nusmv.syncarb5^2.B_0.pkl"
  #data = "../dataset/train/eijk.S208o.S_0.pkl"
  #data = "../dataset/train/ken.flash^12.C_5.pkl"
  #data = "../dataset/tmp/generalize_adj_matrix/adj_ken.flash^12.C_5.pkl"

  # TODO: make val part works here
  train = None
  # train, val = None, None

  if args.train_file is not None:
    with open(args.train_file,'rb') as f:
      train = pickle.load(f)

  #TODO: dump the data in data_gen.py
  net = NeuroPredessor()
  #net = net.cuda() #TODO: modify to accept both CPU and GPU version

  # task_name = args.task_name + '_sr' + str(args.min_n) + 'to' + str(args.max_n) + '_ep' + str(
  #   args.epochs) + '_nr' + str(args.n_rounds) + '_d' + str(args.dim)
  log_file = open(os.path.join(args.log_dir, args.task_name + '.log'), 'a+')
  detail_log_file = open(os.path.join(args.log_dir, args.task_name + '_detail.log'), 'a+')

  loss_fn = nn.BCELoss()
  optim = optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10)
  sigmoid  = nn.Sigmoid()

  best_acc = 0.0
  start_epoch = 0
  end_epoch = 120

  #one batch one iteration at first?
  for epoch in range(start_epoch, end_epoch):

    print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc))
    print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), file=log_file, flush=True)
    print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), file=detail_log_file, flush=True)
    '''
    -------------------------------------------------train--------------------------------------------------------------
    '''
    train_bar = tqdm(train)
    TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
    net.train()
    #FIXME: The train here should be a list contains serverals files (>1)
    for _, prob in enumerate(train_bar):
      optim.zero_grad()
      outputs = net(prob)
      target = torch.Tensor(prob.gt).cuda().float() #TODO: update the loss function here
      outputs = sigmoid(outputs)
      loss = loss_fn(outputs, target)
      desc = 'loss: %.4f; ' % (loss.item())

      loss.backward()
      optim.step()

      preds = torch.where(outputs>0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())

      TP += (preds.eq(1) & target.eq(1)).cpu().sum()
      TN += (preds.eq(0) & target.eq(0)).cpu().sum()
      FN += (preds.eq(0) & target.eq(1)).cpu().sum()
      FP += (preds.eq(1) & target.eq(0)).cpu().sum()
      TOT = TP + TN + FN + FP

      desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % ((TP.item()+TN.item())*1.0/TOT.item(), TP.item()*1.0/TOT.item(), TN.item()*1.0/TOT.item(), FN.item()*1.0/TOT.item(), FP.item()*1.0/TOT.item())
      # train_bar.set_description(desc)
      if (_ + 1) % 100 == 0:
        print(desc, file=detail_log_file, flush=True)

    print(desc, file=log_file, flush=True)

    #FIXME: fix this part for doing validation

    '''
        -------------------------------------------------validation----------------------------------------------------------
    
    val_bar = tqdm(val)
    TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
    net.eval()
    for _, prob in enumerate(val_bar):
      optim.zero_grad()
      outputs = net(prob)
      target = torch.Tensor(prob.is_sat).cuda().float()
      # print(outputs.shape, target.shape)
      # print(outputs, target)
      outputs = sigmoid(outputs)
      preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())

      TP += (preds.eq(1) & target.eq(1)).cpu().sum()
      TN += (preds.eq(0) & target.eq(0)).cpu().sum()
      FN += (preds.eq(0) & target.eq(1)).cpu().sum()
      FP += (preds.eq(1) & target.eq(0)).cpu().sum()
      TOT = TP + TN + FN + FP

      desc = 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
      (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() * 1.0 / TOT.item(), TN.item() * 1.0 / TOT.item(),
      FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())
      # val_bar.set_description(desc)
      if (_ + 1) % 100 == 0:
        print(desc, file=detail_log_file, flush=True)
    print(desc, file=log_file, flush=True)

    acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
    torch.save({'epoch': epoch + 1, 'acc': acc, 'state_dict': net.state_dict()},
               os.path.join(args.model_dir, task_name + '_last.pth.tar'))
    if acc >= best_acc:
      best_acc = acc
      torch.save({'epoch': epoch + 1, 'acc': best_acc, 'state_dict': net.state_dict()},
                 os.path.join(args.model_dir, task_name + '_best.pth.tar'))
                 
    '''