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
from sklearn.model_selection import train_test_split
from time import sleep
import z3
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



#TODO: Try small size sample and test accuracy

#TODO: setup the batch size, iteration and epoch
'''
dataset -> contains around 10000 cases
batch -> one .smt2? (one time of generalized predecessor?) 
iteration -> every .smt2
epoch -> len(dataset)/batch -> 10000/1 maybe?
'''

'''
Note: 
For the label of the problem, 
label of the flexiable node -> generalized predecessor
label of the minimum q (q-like) -> inductive generalization
'''


# def refine_data(problem):
#   df_empty = pd.DataFrame(columns=problem.unpack_matrix.columns, index=problem.unpack_matrix.index)
#   df_empty = df_empty.fillna(1)
#   df_empty = df_empty[df_empty.columns.drop(list(df_empty.filter(regex='m_')))]
#   df_empty.columns = df_empty.columns.str.replace(r'n_', '')
#   df_empty.drop(columns=['old_index'],inplace=True)
#   problem.db_gt = problem.db_gt.reindex(columns=problem.db_gt.columns | df_empty.columns)

# def refine_output(problem, output):
#   '''
#   Find out the node not in graph, and re-construct the output after net() return the value
#   '''
#   output_lst = (output.squeeze()).tolist()
#   single_node_index = [] # store the index
#   var_list = list(problem.db_gt)
#   var_list.pop(0) #remove "filename_nextcube"
#   tmp = problem.value_table[~problem.value_table.index.str.contains('m_')]
#   tmp.index = tmp.index.str.replace("n_","")

#   for i, element in enumerate(var_list):
#     if element not in tmp.index.tolist():
#       single_node_index.append(i)

#   for index in single_node_index:
#     output_lst.insert(index,1)

#   return torch.Tensor(output_lst).cuda().float()

# def del_tensor_ele(arr,index):
#     arr1 = arr[0:index]
#     arr2 = arr[index+1:]
#     return torch.cat((arr1,arr2),dim=0)

def refine_cube(problem): #FIXME: This part has issue!
  '''
  Caculate the variable index in the value table (variable in db_gt)
  -> for reducing the output of NN -> inductive generalization 
  '''
  var_index = []
  tmp_lst_all_node = problem.value_table.index.to_list()[problem.n_nodes:]
  tmp_lst_var = list(problem.db_gt)[1:] #sequence is different from the table
  for element in tmp_lst_var:
    var_index.append(tmp_lst_all_node.index('n_'+str(element)))
  return var_index  

def extract_q_like(problem):
  '''
  Caculate the q index in the value table
  -> for reducing the output of NN -> inductive generalization 
  '''
  q_index = []
  tmp_lst_all_node = problem.value_table.index.to_list()[problem.n_nodes:]
  
  constraints = z3.parse_smt2_file(((problem.filename[0].split('/')[-1]).replace('.pkl','.smt2')).replace('adj_',''), sorts={}, decls={})
  ig_q = constraints[:-1]

  for element in ig_q: # literals in q (in inductive generalization process)
    q_index.append(tmp_lst_all_node.index('n_'+str(element)))
  return q_index  

def refine_target(problem):
  '''
  Refine the is_flexiable -> ignore the single node in the graph
  -> generalized predecessor
  '''
  single_node_index = [] # store the index
  var_list = list(problem.db_gt)
  var_list.pop(0) #remove "filename_nextcube"
  tmp = problem.value_table[~problem.value_table.index.str.contains('m_')]
  tmp.index = tmp.index.str.replace("n_","")

  for i, element in enumerate(var_list):
    if element not in tmp.index.tolist():
      single_node_index.append(i)

  problem.label = [e[1] for e in enumerate(problem.label) if e[0] not in single_node_index]
  # for index in reversed(single_node_index):
  #   problem.label.pop(index)


if __name__ == "__main__":

  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  # Set up the tensorboard log directory
  datetime_str = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
  writer = SummaryWriter('../log/tensorboard'+'-'+datetime_str.replace(' ','_'))

  args = parser.parse_args(['--task-name', 'neuropdr_'+datetime_str.replace(' ','_'), '--dim', '128', '--n_rounds', '512', \
                            '--epochs', '256', \
                            '--gen_log', '../log/data_maker_sr3t10.log', \
                            # '--train-file', '../dataset/GP2graph/train/',\
                            # '--val-file','../dataset/GP2graph/validate/',\
                            '--train-file', '../dataset/IG2graph/train/',\
                            '--val-file','../dataset/IG2graph/validate/',\
                            '--mode', 'test'
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
  
  # Remove the train file which exists bug (has no adj_matrix generated)
  train = [train_file for train_file in train if hasattr(train_file,'adj_matrix')]

  if args.mode=='test':
    train, val = train_test_split(train, test_size=0.2, random_state=42)
  elif args.mode=='train':
    val_lst = walkFile(args.val_file)
    for val_file in val_lst:
      with open(val_file,'rb') as f2:
        val.append(pickle.load(f2))

  #FIXME: This part works strange

  # for train_data in train:
  #   refine_data(train_data)

  #TODO: dump the data in data_gen.py

  # Fetch the index of q for reduce the output of NN
  # if train is not None: q_index = refine_cube(train[0])
  
  net = NeuroPredessor(args)
  net = net.cuda() #TODO: modify to accept both CPU and GPU version

  # task_name = args.task_name + '_sr' + str(args.min_n) + 'to' + str(args.max_n) + '_ep' + str(
  #   args.epochs) + '_nr' + str(args.n_rounds) + '_d' + str(args.dim)
  log_file = open(os.path.join(args.log_dir, args.task_name + '.log'), 'a+')
  detail_log_file = open(os.path.join(args.log_dir, args.task_name + '_detail.log'), 'a+')

  #loss_fn = nn.BCELoss() #TODO: Try to modify this part
  #loss_fn = nn.SmoothL1Loss()
  #loss_fn = nn.CrossEntropyLoss()
  loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10]).cuda())
  #optim = optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10)
  optim = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-10)
  #optim = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-10)
  #optim = optim.Adam(net.parameters(), lr=0.0002, weight_decay=1e-10) #TODO: Try to figure out what parameter is optimal
  sigmoid  = nn.Sigmoid()

  best_acc = 0.0
  best_precision = 0.0
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
    best_precision = model['precision']
    net.load_state_dict(model['state_dict'])

  iteration = 0
  #one batch one iteration at first?
  for epoch in range(start_epoch, args.epochs):
    print('==> %d/%d epoch, previous best precision: %.3f' % (epoch+1, args.epochs, best_precision))
    print('==> %d/%d epoch, previous best accuracy: %.3f' % (epoch+1, args.epochs, best_acc))
    print('==> %d/%d epoch, previous best precision: %.3f' % (epoch+1, args.epochs, best_precision), file=log_file, flush=True)
    print('==> %d/%d epoch, previous best accuracy: %.3f' % (epoch+1, args.epochs, best_acc), file=log_file, flush=True)
    print('==> %d/%d epoch, previous best precision: %.3f' % (epoch+1, args.epochs, best_precision), file=detail_log_file, flush=True)
    print('==> %d/%d epoch, previous best accuracy: %.3f' % (epoch+1, args.epochs, best_acc), file=detail_log_file, flush=True)
    '''
    -------------------------------------------------train--------------------------------------------------------------
    '''
    train_bar = tqdm(train)
    TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
    net.train()
    perfection_rate = 0 # Used to record the perfection ratio of the validation set
    all = 0 # Used to record the number of all samples in the validation set
    #FIXME: The train here should be a list contains serverals files (>1)
    for _, prob in enumerate(train_bar):
      iteration += 1
      q_index = refine_cube(prob)
      optim.zero_grad()
      outputs = net(prob)
      target = torch.Tensor(prob.label).cuda().float() #TODO: update the loss function here
      outputs = sigmoid(outputs)
      
      torch_select = torch.Tensor(q_index).cuda().int() 
      outputs = torch.index_select(outputs, 0, torch_select)
      
      #outputs = refine_output(prob,outputs)
      loss = loss_fn(outputs, target)
      desc = 'loss: %.4f; ' % (loss.item())

      loss.backward()
      optim.step()

      preds = torch.where(outputs>0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())

      # Calulate the perfect accuracy
      all = all + 1
      if target.equal(preds): perfection_rate = perfection_rate + 1
      
      TP += (preds.eq(1) & target.eq(1)).cpu().sum()
      TN += (preds.eq(0) & target.eq(0)).cpu().sum()
      FN += (preds.eq(0) & target.eq(1)).cpu().sum()
      FP += (preds.eq(1) & target.eq(0)).cpu().sum()
      TOT = TP + TN + FN + FP

      desc += 'perfection rate: %.3f, acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (perfection_rate*1.0/all,(TP.item()+TN.item())*1.0/TOT.item(), TP.item()*1.0/TOT.item(), TN.item()*1.0/TOT.item(), FN.item()*1.0/TOT.item(), FP.item()*1.0/TOT.item())
      writer.add_scalar('accuracy/train_perfection_ratio', (perfection_rate*1.0/all)*100, iteration)
      
      writer.add_scalar('confusion_matrix/true_possitive', TP.item()*1.0/TOT.item(), iteration)
      writer.add_scalar('confusion_matrix/false_possitive', FP.item()*1.0/TOT.item(), iteration)
      try:
        writer.add_scalar('confusion_matrix/precision', TP.item()*1.0/(TP.item()*1.0 + FP.item()*1.0), iteration)
      except:
        writer.add_scalar('confusion_matrix/precision', 0, iteration)
      writer.add_scalar('model_evaluate/TPR_recall',  TP.item()*1.0/(TP.item()*1.0 + FN.item()*1.0), iteration)
      writer.add_scalar('model_evaluate/FPR',  FP.item()*1.0/(FP.item()*1.0+ TN.item()*1.0), iteration)
      # Plot More Metric Curve on tensorboard
      # writer.add_scalar('model_evaluate/MCC',  (TP.item()*1.0 - FP.item()*1.0)*1.0/(TP.item()*1.0 + FP.item()*1.0 + FN.item()*1.0 - TP.item()*1.0 - FP.item()*1.0 - FN.item()*1.0), iteration)
      # writer.add_scalar('model_evaluate/MSE',  (TP.item()*1.0 + TN.item()*1.0 - FP.item()*1.0 - FN.item()*1.0)*1.0/(TP.item()*1.0 + TN.item()*1.0 + FP.item()*1.0 + FN.item()*1.0), iteration)
      # writer.add_scalar('model_evaluate/RMSE',  (TP.item()*1.0 + TN.item()*1.0 - FP.item()*1.0 - FN.item()*1.0)*1.0/(TP.item()*1.0 + TN.item()*1.0 + FP.item()*1.0 + FN.item()*1.0), iteration)
      # writer.add_scalar('model_evaluate/MAE',  (TP.item()*1.0 + TN.item()*1.0 - FP.item()*1.0 - FN.item()*1.0)*1.0/(TP.item()*1.0 + TN.item()*1.0 + FP.item()*1.0 + FN.item()*1.0), iteration)
      # writer.add_scalar('model_evaluate/MAD',  (TP.item()*1.0 + TN.item()*1.0 - FP.item()*1.0 - FN.item()*1.0)*1.0/(TP.item()*1.0 + TN.item()*1.0 + FP.item()*1.0 + FN.item()*1.0), iteration)
      # writer.add_scalar('model_evaluate/MAPE',  (TP.item()*1.0 + TN.item()*1.0 - FP.item()*1.0 - FN.item()*1.0)*1.0/(TP.item()*1.0 + TN.item()*1.0 + FP.item()*1.0 + FN.item()*1.0), iteration)
      # writer.add_scalar('model_evaluate/MSPE',  (TP.item()*1.0 + TN.item()*1.0 - FP.item()*1.0 - FN.item()*1.0)*1.0/(TP.item()*1.0 + TN.item()*1.0 + FP.item()*1.0 + FN.item()*1.0), iteration)
      # writer.add_scalar('model_evaluate/MASE',  (TP.item()*1.0 + TN.item()*1.0 - FP.item()*1.0 - FN.item()*1.0)*1.0/(TP.item()*1.0 + TN.item()*1.0 + FP.item()*1.0 + FN.item()*1.0), iteration)
      # writer.add_scalar('model_evaluate/ROC_curve', (TP.item()*1.0/(TP.item()*1.0+ FN.item()*1.0)), (FP.item()*1.0/(FP.item()*1.0+TN.item()*1.0)))
      
      writer.add_scalar('model_evalute/F1-Socre', 2*TP.item()*1.0/(2*TP.item()*1.0+FP.item()*1.0+FN.item()*1.0), iteration)

      # train_bar.set_description(desc)
      if (_ + 1) % 100 == 0:
        print(desc, file=detail_log_file, flush=True)

    print(desc, file=log_file, flush=True)
    writer.add_scalar('accuracy/training_accuracy', (TP.item()+TN.item())*1.0/TOT.item(), epoch)
    writer.add_scalar('loss/training_loss', loss.item(), epoch)
    

    #FIXME: fix this part for doing validation

    '''
        -------------------------------------------------validation----------------------------------------------------------
    '''

    val_bar = tqdm(val)
    TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
    net.eval()
    perfection_rate = 0 # Used to record the perfection ratio of the validation set
    all = 0 # Used to record the number of all samples in the validation set
    for _, prob in enumerate(val_bar):
      q_index = refine_cube(prob)
      optim.zero_grad()
      outputs = net(prob)
      target = torch.Tensor(prob.label).cuda().float()
      # print(outputs.shape, target.shape)
      # print(outputs, target)
      outputs = sigmoid(outputs)
      torch_select = torch.Tensor(q_index).cuda().int()
      outputs = torch.index_select(outputs, 0, torch_select)
      preds = torch.where(outputs > 0.997, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())

      # Calulate the perfect accuracy
      all = all + 1
      if target.equal(preds): perfection_rate = perfection_rate + 1

      TP += (preds.eq(1) & target.eq(1)).cpu().sum()
      TN += (preds.eq(0) & target.eq(0)).cpu().sum()
      FN += (preds.eq(0) & target.eq(1)).cpu().sum()
      FP += (preds.eq(1) & target.eq(0)).cpu().sum()
      TOT = TP + TN + FN + FP

      desc = 'perfection rate: %.3f, acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
      perfection_rate*1.0/all,
      (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() * 1.0 / TOT.item(), TN.item() * 1.0 / TOT.item(),
      FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())

      writer.add_scalar('accuracy/predict_perfection_ratio', (perfection_rate*1.0/all)*100, iteration)
      
      # val_bar.set_description(desc)
      if (_ + 1) % 100 == 0:
        print(desc, file=detail_log_file, flush=True)
    print(desc, file=log_file, flush=True)

    acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
    try:
      val_precision = TP.item()*1.0/(TP.item()*1.0 + FP.item()*1.0)
    except:
      val_precision = 0
    
    writer.add_scalar('accuracy/validation_accuracy', acc, epoch)
    torch.save({'epoch': epoch + 1, 'acc': acc, 'precision': val_precision, 'state_dict': net.state_dict()},
               os.path.join(args.model_dir, args.task_name + '_last.pth.tar'))
    if val_precision >= best_precision:
      best_precision = val_precision
      torch.save({'epoch': epoch + 1, 'acc': acc, 'precision': best_precision, 'state_dict': net.state_dict()},
                 os.path.join(args.model_dir, args.task_name + '_best_precision.pth.tar'))
      
    if acc >= best_acc:
      best_acc = acc
      # torch.save({'epoch': epoch + 1, 'acc': best_acc, 'precision': val_precision, 'state_dict': net.state_dict()},
      #            os.path.join(args.model_dir, args.task_name + '_best.pth.tar'))
        
  try:
    writer.close()
  except BaseException:
    writer.close()
