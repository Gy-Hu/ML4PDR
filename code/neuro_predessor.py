import torch
import torch.nn as nn

from mlp import MLP


class NeuroPredessor(nn.Module):
    def __init__(self):
        super(NeuroPredessor, self).__init__()
        self.dim = 128

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.var_init = nn.Linear(1,self.dim) # include v and m
        self.node_init = nn.Linear(1, self.dim)  # initialize a vector (1xd)

        self.var_msg = MLP(self.dim, self.dim, self.dim)
        self.node_msg = MLP(self.dim, self.dim, self.dim)

        self.var_update = nn.LSTM(self.dim, self.dim)
        self.node_update = nn.LSTM(self.dim, self.dim)

        self.node_vote = MLP(self.dim, self.dim, 1)
        self.denom = torch.sqrt(torch.Tensor([self.dim]))

    def forward(self, problem):
        n_var = 5 #TODO: Refine here (modify in data_gen.py)
        n_node = 10 #TODO: Refine here (modify in data_gen.py)
        ts_var_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()
        unpack = torch.sparse.FloatTensor(ts_var_unpack_indices, torch.ones(problem.n_cells),
                                          torch.Size([n_var, n_node])).to_dense().cuda()
        init_ts = self.init_ts.cuda()
        var_init = self.var_init(init_ts).view(1, 1, -1) # encode true or false here
        node_init = self.node_init(init_ts).view(1, 1, -1) # re-construct the dimension, size = [1, 1, 128]
        var_init = var_init.repeat(1, n_var, 1)
        node_init = node_init.repeat(1, n_node, 1)

        var_state = (var_init, torch.zeros(1, n_var, self.dim).cuda()) # resize for LSTM
        node_state = (node_init, torch.zeros(1, n_node, self.dim).cuda()) #resize for LSTM
        # adj_martix initialize here

        # message passing procedure
        for _ in range(self.args.n_rounds): #TODO: Using LSTM to eliminate the error brought by symmetry
            var_hidden = var_state[0].squeeze(0)  # initialize to zero
            var_pre_msg = self.var_msg(var_hidden)
            child_to_par_msg = torch.matmul(unpack.t(), var_pre_msg) #TODO: ask question "two embedding of m here"
            _, node_state = self.var_update(child_to_par_msg.unsqueeze(0), node_state)  # Ch (t+1) , C(t+1)= Cu ([Ch(t), M.t*Lmsg(L(t)) ])

            node_hidden = node_state[0].squeeze(0)
            node_pre_msg = self.node_msg(node_hidden)
            par_to_child_msg = torch.matmul(unpack, node_pre_msg)
            _, var_state = self.node_update(par_to_child_msg[0].unsqueeze(0), var_state)  # Lh(t+1) ,L(t+1)= Lu ([Lh(t), Flip(L(t)), M.t*Cmsg(C(t+1)) ])

        logits = var_state[0].squeeze(0)
        #TODO: update here with the correct number
        vote = self.node_vote(logits[:(n_var-n_node) , :]) # (a+b) * dim -> a * dim
        vote_mean = torch.mean(vote, dim=1)
        return vote_mean



