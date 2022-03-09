import argparse
import pickle
import os
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from config import parser
from neuro_predessor import NeuroPredessor
from data_gen import problem, walkFile
import pandas as pd
from pathlib import Path

#TODO: Try small size sample and test accuracy

#TODO: setup the batch size, iteration and epoch
'''
dataset -> contains around 10000 cases
batch -> one .smt2? (one time of generalized predecessor?) 
iteration -> every .smt2
epoch -> len(dataset)/batch -> 10000/1 maybe?
'''

def refine_data(problem):
  df_empty = pd.DataFrame(columns=problem.unpack_matrix.columns, index=problem.unpack_matrix.index)
  df_empty = df_empty.fillna(1)
  df_empty = df_empty[df_empty.columns.drop(list(df_empty.filter(regex='m_')))]
  df_empty.columns = df_empty.columns.str.replace(r'n_', '')
  df_empty.drop(columns=['old_index'],inplace=True)
  problem.db_gt = problem.db_gt.reindex(columns=problem.db_gt.columns | df_empty.columns)

def refine_output(problem, output):
  '''
  Find out the node not in graph, and re-construct the output after net() return the value
  '''
  output_lst = (output.squeeze()).tolist()
  single_node_index = [] # store the index
  var_list = list(problem.db_gt)
  var_list.pop(0) #remove "filename_nextcube"
  tmp = problem.value_table[~problem.value_table.index.str.contains('m_')]
  tmp.index = tmp.index.str.replace("n_","")

  for i, element in enumerate(var_list):
    if element not in tmp.index.tolist():
      single_node_index.append(i)

  for index in single_node_index:
    output_lst.insert(index,1)

  return torch.Tensor(output_lst).cuda().float()

def refine_target(problem):
  '''
  Refine the is_flexiable
  '''
  single_node_index = [] # store the index
  var_list = list(problem.db_gt)
  var_list.pop(0) #remove "filename_nextcube"
  tmp = problem.value_table[~problem.value_table.index.str.contains('m_')]
  tmp.index = tmp.index.str.replace("n_","")

  for i, element in enumerate(var_list):
    if element not in tmp.index.tolist():
      single_node_index.append(i)

  for index in reversed(single_node_index):
    problem.is_flexible.pop(index)


if __name__ == "__main__":

  args = parser.parse_args(['--task-name', 'neuropdr_no1', '--dim', '128', '--n_rounds', '120', \
                            '--epochs', '20', \
                            '--gen_log', '/home/gary/coding_env/NeuroSAT/log/data_maker_sr3t10.log', \
                            '--train-file', '../dataset/train/',\
                            '--val-file','../dataset/eval/'
                            ])


  #TODO: refine the ground truth (which is the is_flexible[] list) with MUST here
  #data = "../dataset/train/nusmv.syncarb5^2.B_0.pkl"
  #data = "../dataset/train/eijk.S208o.S_0.pkl"
  #data = "../dataset/train/ken.flash^12.C_5.pkl"
  #data = "../dataset/tmp/generalize_adj_matrix/adj_ken.flash^12.C_5.pkl"

  # TODO: make val part works here
  train = []
  val = []
  # train, val = None, None



  if args.train_file is not None:
    #print(sys.path[0])
    #print(os.getcwd())
    train_lst = walkFile(args.train_file)
    for train_file in train_lst:
      with open(train_file,'rb') as f:
        train.append(pickle.load(f))

  eval_lst = walkFile(args.val_file)
  for val_file in eval_lst:
    with open(val_file,'rb') as f2:
      val.append(pickle.load(f2))

  #FIXME: This part works strange

  # for train_data in train:
  #   refine_data(train_data)

  #TODO: dump the data in data_gen.py
  net = NeuroPredessor(args)
  net = net.cuda() #TODO: modify to accept both CPU and GPU version

  # task_name = args.task_name + '_sr' + str(args.min_n) + 'to' + str(args.max_n) + '_ep' + str(
  #   args.epochs) + '_nr' + str(args.n_rounds) + '_d' + str(args.dim)
  log_file = open(os.path.join(args.log_dir, args.task_name + '.log'), 'a+')
  detail_log_file = open(os.path.join(args.log_dir, args.task_name + '_detail.log'), 'a+')

  loss_fn = nn.BCELoss() #TODO: Try to modify this part
  optim = optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10)
  #optim = optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-10) #TODO: Try to figure out what parameter is optimal
  sigmoid  = nn.Sigmoid()

  best_acc = 0.0
  start_epoch = 0
  #end_epoch = 120

  if train is not None:
    for problem in train:
      refine_target(problem)
    print('num of train batches: ', len(train), file=log_file, flush=True)

  if val is not None:
    for val_file in val:
      refine_target(val_file)

  if args.restore is not None:
    print('restoring from', args.restore, file=log_file, flush=True)
    model = torch.load(args.restore)
    start_epoch = model['epoch']
    best_acc = model['acc']
    net.load_state_dict(model['state_dict'])

  #one batch one iteration at first?
  for epoch in range(start_epoch, args.epochs):

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
      target = torch.Tensor(prob.is_flexible).cuda().float() #TODO: update the loss function here
      outputs = sigmoid(outputs)
      #outputs = refine_output(prob,outputs)
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
    '''

    val_bar = tqdm(val)
    TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
    net.eval()
    for _, prob in enumerate(val_bar):
      optim.zero_grad()
      outputs = net(prob)
      target = torch.Tensor(prob.is_flexible).cuda().float()
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
               os.path.join(args.model_dir, args.task_name + '_last.pth.tar'))
    if acc >= best_acc:
      best_acc = acc
      torch.save({'epoch': epoch + 1, 'acc': best_acc, 'state_dict': net.state_dict()},
                 os.path.join(args.model_dir, args.task_name + '_best.pth.tar'))
