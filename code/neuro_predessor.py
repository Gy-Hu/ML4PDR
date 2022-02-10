import torch
import torch.nn as nn
import z3

from mlp import MLP


class NeuroPredessor(nn.Module):
    def __init__(self):
        super(NeuroPredessor, self).__init__()
        self.dim = 128

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        #TODO: Using 2 separated nn.Linear to do literals embedding
        self.true_init = nn.Linear(1,self.dim) #for node return true
        self.false_init = nn.Linear(1, self.dim) #for node return false

        self.children_msg = MLP(self.dim, self.dim, self.dim) #for children to pass message
        self.parent_msg = MLP(self.dim, self.dim, self.dim) #for parents to pass message

        self.node_update = nn.LSTM(self.dim, self.dim) #update node (exclude variable)
        self.var_update = nn.LSTM(self.dim, self.dim) #udpate variable and node

        #FIXME: fix here (how to defind new vote?)
        self.var_vote = MLP(self.dim, self.dim, 1) #vote for variable and node
        self.denom = torch.sqrt(torch.Tensor([self.dim]))

    def forward(self, problem):
        n_var = problem.n_vars #TODO: Refine here (modify in data_gen.py)
        n_node = problem.n_nodes #TODO: Refine here (modify in data_gen.py)
        # ts_var_unpack_indices = torch.Tensor(problem.adj_matrix).t().long() #TODO: refine the adj matrix here
        # unpack = torch.sparse.FloatTensor(ts_var_unpack_indices, torch.ones(problem.n_cells), #TODO: refine the n_cells.. here
                                          # torch.Size([n_var, n_node])).to_dense().cuda()
        unpack = problem.adj_matrix
        init_ts = self.init_ts.cuda()

        # TODO: change the init part to true/false init
        dict_vt = dict(zip(problem.value_table['index'], problem.value_table['Value']))

        for unknown in problem.unknown:
            if unknown is True:
                unknown = self.true_init(init_ts).view(1, 1, -1) #<-assign true init tensor
            else:
                unknown = self.false_init(init_ts).view(1, 1, -1) #<-assign false init tensor

        all_init = torch.cat(self.true_init, self.false_init)

        # var_init = self.var_init(init_ts).view(1, 1, -1) # encode true or false here
        # node_init = self.node_init(init_ts).view(1, 1, -1) # re-construct the dimension, size = [1, 1, 128]
        # var_init = var_init.repeat(1, n_var, 1)
        # node_init = node_init.repeat(1, n_node, 1)

        var_state = (all_init[:], torch.zeros(1, n_var, self.dim).cuda()) # resize for LSTM, (ht, ct)
        '''
        var_state[:] -> all node includes input, input_prime, variable
        var_state[?:] -> node exclude input, input_prime, variable
        var_state[:?] -> only input, input_prime, variable (without m node)
        '''

        # adj_martix initialize here

        # message passing procedure
        #TODO: refine the n_rounds
        for _ in range(self.args.n_rounds): #TODO: Using LSTM to eliminate the error brought by symmetry

            var_pre_msg = self.children_msg(var_state[:][0].squeeze(0))
            child_to_par_msg = torch.matmul(unpack.t(), var_pre_msg) #TODO: ask question "two embedding of m here"
            _, var_state[unknown:] = self.var_update(child_to_par_msg.unsqueeze(0), var_state[unknown:])  #TODO: replace node_state with the partial var_state

            node_pre_msg = self.parent_msg(var_state[unknown:][0].squeeze(0))
            par_to_child_msg = torch.matmul(unpack, node_pre_msg)
            _, var_state[:] = self.node_update(par_to_child_msg[0].unsqueeze(0), var_state[:])

        logits = var_state[0].squeeze(0)
        #TODO: update here with the correct number
        vote = self.node_vote(logits[:(n_var-n_node) , :]) # (a+b) * dim -> a * dim
        vote_mean = torch.mean(vote, dim=1)
        return vote_mean



