import argparse
import pickle
import os

from zmq import device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from config import parser
from neuro_ig_no_enumerate import NeuroPredessor
from data_gen import problem, walkFile
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from time import sleep
import z3
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

class ReloadedInt(int):
    def __truediv__(self, other):
        if other == 0:
            return 0
        else:
            return super().__truediv__(other)
    def __rtruediv__(self,other):
        if other == 0:
            return 0
        else:
            return super().__truediv__(other)

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(BCEFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.sum()

class WeightedBCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=8, reduction='sum'):
        super(WeightedBCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        logits = torch.sigmoid(logits)
        logits = torch.clamp(logits, min=1e-7, max=1-1e-7)
        loss = - self.pos_weight * target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class GraphDataset(Dataset):
    def __init__(self,data_root):
        self.data_root = data_root
        self.samples = []
        self.__init_dataset()
        self.__remove_adj_null_file()
        self.__refine_target_and_output()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prob_main_info = {
            'n_vars' : self.samples[idx].n_vars,
            'n_nodes' : self.samples[idx].n_nodes,
            'unpack' : (torch.from_numpy(self.samples[idx].adj_matrix.astype(np.float32).values)).to('cuda'),
            'refined_output' : self.samples[idx].refined_output,
            'label' : self.samples[idx].label
        }
        dict_vt = dict(zip((self.samples[idx].value_table).index, (self.samples[idx].value_table).Value))
        return prob_main_info, dict_vt
    
    def __init_dataset(self):
        train_lst = walkFile(self.data_root)
        for train_file in train_lst[:]:
            with open(train_file, 'rb') as f:
                self.samples.append(pickle.load(f))

    def __refine_target_and_output(self):
        for problem in self.samples:
            
            var_list = list(problem.db_gt)
            var_list.pop(0)  # remove "filename_nextcube"
            tmp = problem.value_table[~problem.value_table.index.str.contains('m_')]
            tmp.index = tmp.index.str.replace("n_", "")

            single_node_index = []  # store the index
            for i, element in enumerate(var_list):
                if element not in tmp.index.tolist():
                    single_node_index.append(i)

            problem.label = [e[1] for e in enumerate(
                problem.label) if e[0] not in single_node_index]
            
            # assert the label will not be all zero
            assert(sum(problem.label) != 0)

            '''
            Finish refine the target, now try to refine the output 
            '''
            var_index = [] # Store the index that is in the graph and in the ground truth table
            tmp_lst_var = list(problem.db_gt)[1:]
            # The groud truth we need to focus on
            focus_gt = [e[1] for e in enumerate(tmp_lst_var) if e[0] not in single_node_index]
            # Try to fetch the index of the variable in the value table (variable in db_gt)
            tmp_lst_all_node = problem.value_table.index.to_list()[problem.n_nodes:]
            for element in focus_gt:
                var_index.append(tmp_lst_all_node.index('n_'+str(element)))
            problem.refined_output = var_index
            assert(len(problem.refined_output) == len(problem.label))
        #print('num of train batches: ', len(train), file=log_file, flush=True)
    
    def __remove_adj_null_file(self):
        # Remove the train file which exists bug (has no adj_matrix generated)
        self.samples = [train_file for train_file in self.samples if hasattr(train_file, 'adj_matrix')]

def collate_wrapper(batch):
    prob_main_info, dict_vt = zip(*batch)
    return prob_main_info, dict_vt

if __name__ == "__main__":
    device = 'cuda'
    datetime_str = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    

    args = parser.parse_args(['--task-name', 'neuropdr_'+datetime_str.replace(' ', '_'), '--dim', '128', '--n_rounds', '512',
                              '--epochs', '256',
                              #'--log-dir', str(Path(__file__).parent.parent /'log/tmp/'), \
                              '--train-file', '../dataset/IG2graph/train_no_enumerate/',\
                              '--val-file', '../dataset/IG2graph/validate_no_enumerate/',\
                              '--mode', 'test'
                              ])

    if args.mode == 'debug':
        writer = SummaryWriter('../log/tmp/tensorboard'+'-' + datetime_str.replace(' ', '_'))
    elif args.mode == 'test':
        writer = SummaryWriter('../log/tensorboard'+'-' + datetime_str.replace(' ', '_'))

    all_train = []
    train = []
    val = []

    all_graph = GraphDataset(args.train_file)

    
    if args.mode == 'test' or args.mode == 'debug':
        train_size = int(0.6 * len(all_graph))
        validation_size = int(0.2 * len(all_graph))
        test_size = len(all_graph) - train_size - validation_size

        # Randomly
        train_dataset, test_dataset = torch.utils.data.random_split(all_graph, [train_size + validation_size, test_size])
        _ , validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])

        # Sequentially
        #train_dataset = torch.utils.data.Subset(all_graph, range(train_size))
        #validation_dataset = torch.utils.data.Subset(all_graph, range(train_size, train_size + validation_size))
        #test_dataset = torch.utils.data.Subset(all_graph, range(validation_size, len(all_graph)))

        train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=0)

        validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=0)

        test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=0)

    net = NeuroPredessor(args)
    net = net.to('cuda')  # TODO: modify to accept both CPU and GPU version
    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net)

    log_file = open(os.path.join(args.log_dir, args.task_name + '.log'), 'a+')
    detail_log_file = open(os.path.join(
        args.log_dir, args.task_name + '_detail.log'), 'a+')

    #loss_fn = nn.BCELoss(reduction='sum')
    #loss_fn = BCEFocalLoss()
    loss_fn = WeightedBCELosswithLogits()
    optim = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-10)
    sigmoid = nn.Sigmoid()

    best_acc = 0.0
    best_precision = 0.0
    start_epoch = 0

    if args.restore is not None:
        print('restoring from', args.restore, file=log_file, flush=True)
        model = torch.load(args.restore)
        start_epoch = model['epoch']
        best_acc = model['acc']
        best_precision = model['precision']
        net.load_state_dict(model['state_dict'])

    iteration = 0
    # one batch one iteration at first?
    for epoch in range(start_epoch, args.epochs):
        print('==> %d/%d epoch, previous best precision: %.3f' %
              (epoch+1, args.epochs, best_precision))
        print('==> %d/%d epoch, previous best accuracy: %.3f' %
              (epoch+1, args.epochs, best_acc))
        print('==> %d/%d epoch, previous best precision: %.3f' %
              (epoch+1, args.epochs, best_precision), file=log_file, flush=True)
        print('==> %d/%d epoch, previous best accuracy: %.3f' %
              (epoch+1, args.epochs, best_acc), file=log_file, flush=True)
        print('==> %d/%d epoch, previous best precision: %.3f' % (epoch+1,
              args.epochs, best_precision), file=detail_log_file, flush=True)
        print('==> %d/%d epoch, previous best accuracy: %.3f' %
              (epoch+1, args.epochs, best_acc), file=detail_log_file, flush=True)
        '''
        -----------------------train----------------------------------
        '''
        train_bar = tqdm(train_loader)
        TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(
            1).long(), torch.zeros(1).long(), torch.zeros(1).long()
        net.train()
        perfection_rate = 0  # Used to record the perfection ratio of the validation set
        all = 0  # Used to record the number of all samples in the validation set
        # FIXME: The train here should be a list contains serverals files (>1)
        for batch_index, (prob,vt_dict) in enumerate(train_bar):
            # _ indicates batch index here
            assert(len(prob) == len(vt_dict))
            # traverse the batch
            loss = torch.zeros(1).to(device)
            for prob_index in range(len(prob)):
                iteration += 1
                q_index = prob[prob_index]['refined_output']
                optim.zero_grad()
                outputs = net((prob[prob_index],vt_dict[prob_index]))
                # TODO: update the loss function here
                target = torch.Tensor(prob[prob_index]['label']).to('cuda').float()
                #outputs = sigmoid(outputs)

                torch_select = torch.Tensor(q_index).to('cuda').int()
                outputs = torch.index_select(outputs, 0, torch_select)

                #outputs = refine_output(prob, outputs)
                this_loss = loss_fn(outputs, target)
                desc = 'loss: %.4f; ' % (this_loss.item())
                loss = loss+this_loss

            loss.backward()
            optim.step()

            preds = torch.where(outputs > 0.5, torch.ones(
                outputs.shape).to('cuda'), torch.zeros(outputs.shape).to('cuda'))

            # Calulate the perfect accuracy
            all = all + 1
            if target.equal(preds):
                perfection_rate = perfection_rate + 1

            TP += (preds.eq(1) & target.eq(1)).cpu().sum()
            TN += (preds.eq(0) & target.eq(0)).cpu().sum()
            FN += (preds.eq(0) & target.eq(1)).cpu().sum()
            FP += (preds.eq(1) & target.eq(0)).cpu().sum()
            TOT = TP + TN + FN + FP

            desc += 'perfection rate: %.3f, acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (perfection_rate*1.0/all, (TP.item()+TN.item(
            ))*1.0/TOT.item(), TP.item()*1.0/TOT.item(), TN.item()*1.0/TOT.item(), FN.item()*1.0/TOT.item(), FP.item()*1.0/TOT.item())
            writer.add_scalar('accuracy/train_perfection_ratio',
                              (perfection_rate*1.0/all)*100, iteration)

            writer.add_scalar('confusion_matrix/true_possitive',
                              TP.item()*1.0/TOT.item(), iteration)
            writer.add_scalar('confusion_matrix/false_possitive',
                              FP.item()*1.0/TOT.item(), iteration)

            writer.add_scalar('confusion_matrix/precision', ReloadedInt(TP.item()* 1.0)/(TP.item()*1.0 + FP.item()*1.0), iteration)
            writer.add_scalar('model_evaluate/TPR_recall',  ReloadedInt(TP.item()* 1.0)/(TP.item()*1.0 + FN.item()*1.0), iteration)
            writer.add_scalar('model_evaluate/FPR',  ReloadedInt(FP.item() * 1.0)/(FP.item()*1.0 + TN.item()*1.0), iteration)
            writer.add_scalar('model_evalute/F1-Socre', ReloadedInt(2*TP.item()*1.0) / (2*TP.item()*1.0+FP.item()*1.0+FN.item()*1.0), iteration)
            
            if (batch_index + 1) % 100 == 0:
                print(desc, file=detail_log_file, flush=True)

        print(desc, file=log_file, flush=True)
        writer.add_scalar('accuracy/training_accuracy',
                          (TP.item()+TN.item())*1.0/TOT.item(), epoch)
        writer.add_scalar('loss/training_loss', loss.item(), epoch)

        '''
        -------------------------validation--------------------------------
        '''

        val_bar = tqdm(validation_loader)
        TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
        net.eval()
        perfection_rate = 0  # Used to record the perfection ratio of the validation set
        all = 0  # Used to record the number of all samples in the validation set
        for batch_index, (prob,vt_dict) in enumerate(val_bar):
            q_index = prob[0]['refined_output']
            optim.zero_grad()
            outputs = net((prob[0],vt_dict[0]))
            target = torch.Tensor(prob[0]['label']).to('cuda').float()
            #outputs = sigmoid(outputs)
            torch_select = torch.Tensor(q_index).to('cuda').int()
            outputs = torch.index_select(outputs, 0, torch_select)
            preds = torch.where(outputs > 0.5, torch.ones(
                outputs.shape).to('cuda'), torch.zeros(outputs.shape).to('cuda'))

            loss = loss_fn(outputs, target)

            # Calulate the perfect accuracy
            all = all + 1
            if target.equal(preds):
                perfection_rate = perfection_rate + 1

            TP += (preds.eq(1) & target.eq(1)).cpu().sum()
            TN += (preds.eq(0) & target.eq(0)).cpu().sum()
            FN += (preds.eq(0) & target.eq(1)).cpu().sum()
            FP += (preds.eq(1) & target.eq(0)).cpu().sum()
            TOT = TP + TN + FN + FP

            desc = 'validation loss: %.3f, perfection rate: %.3f, acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
                loss,
                perfection_rate*1.0/all,
                (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() *
                1.0 / TOT.item(), TN.item() * 1.0 / TOT.item(),
                FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())

            writer.add_scalar('loss/validation_loss', loss.item(), epoch)

            writer.add_scalar('accuracy/validation_predict_perfection_ratio',
                              (perfection_rate*1.0/all)*100, iteration)

            # val_bar.set_description(desc)
            if (batch_index + 1) % 100 == 0:
                print(desc, file=detail_log_file, flush=True)
        print(desc, file=log_file, flush=True)

        acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
 
        val_precision = ReloadedInt(TP.item()*1.0)/(TP.item()*1.0 + FP.item()*1.0)


        writer.add_scalar('accuracy/validation_accuracy', acc, epoch)


        '''
        ------------------------testing-----------------------------------
        '''

        test_bar = tqdm(test_loader)
        TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
        net.eval()
        perfection_rate = 0  # Used to record the perfection ratio of the validation set
        all = 0  # Used to record the number of all samples in the validation set
        for batch_index , (prob,vt_dict) in enumerate(test_bar):
            q_index = prob[0]['refined_output']
            optim.zero_grad()
            outputs = net((prob[0],vt_dict[0]))
            target = torch.Tensor(prob[0]['label']).to('cuda').float()
            #outputs = sigmoid(outputs)
            torch_select = torch.Tensor(q_index).to('cuda').int()
            outputs = torch.index_select(outputs, 0, torch_select)
            preds = torch.where(outputs > 0.5, torch.ones(
                outputs.shape).to('cuda'), torch.zeros(outputs.shape).to('cuda'))

            loss = loss_fn(outputs, target)

            # Calulate the perfect accuracy
            all = all + 1
            if target.equal(preds):
                perfection_rate = perfection_rate + 1

            TP += (preds.eq(1) & target.eq(1)).cpu().sum()
            TN += (preds.eq(0) & target.eq(0)).cpu().sum()
            FN += (preds.eq(0) & target.eq(1)).cpu().sum()
            FP += (preds.eq(1) & target.eq(0)).cpu().sum()
            TOT = TP + TN + FN + FP

            desc = 'testing loss: %.3f, perfection rate: %.3f, acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
                loss,
                perfection_rate*1.0/all,
                (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() *
                1.0 / TOT.item(), TN.item() * 1.0 / TOT.item(),
                FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())

            writer.add_scalar('loss/tesing_loss', loss.item(), epoch)

            writer.add_scalar('accuracy/testing_predict_perfection_ratio',
                              (perfection_rate*1.0/all)*100, iteration)

            # val_bar.set_description(desc)
            if (batch_index + 1) % 100 == 0:
                print(desc, file=detail_log_file, flush=True)
        print(desc, file=log_file, flush=True)

        acc = (TP.item() + TN.item()) * 1.0 / TOT.item()

        val_precision = ReloadedInt(TP.item()*1.0)/(TP.item()*1.0 + FP.item()*1.0)

        writer.add_scalar('accuracy/testing_accuracy', acc, epoch)
        torch.save({'epoch': epoch + 1, 'acc': acc, 'precision': val_precision, 'state_dict': net.state_dict()},
                   os.path.join(args.model_dir, args.task_name + '_last.pth.tar'))
        if val_precision >= best_precision:
            best_precision = val_precision
            torch.save({'epoch': epoch + 1, 'acc': acc, 'precision': best_precision, 'state_dict': net.state_dict()},
                       os.path.join(args.model_dir, args.task_name + '_best_precision.pth.tar'))

        if acc >= best_acc:
            best_acc = acc

    try:
        writer.close()
    except BaseException:
        writer.close()
