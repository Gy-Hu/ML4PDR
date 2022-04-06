import torch
import torch.nn as nn

'''
-------------------Simple MLP----------------------

class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.f1 = nn.ReLU()
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.f2 = nn.ReLU()
    self.l3 = nn.Linear(hidden_dim, out_dim)
    self.f3 = nn.Sigmoid() # between 0-1
    #TODO: Modify to non-linear version for var_state

  def forward(self, x):
    x = self.l1(x)
    x = self.f1(x)
    x = self.l2(x)
    x = self.f2(x)
    x = self.l3(x)
    output = self.f3(x)

    return output
'''

'''
-----------------Non-linear MLP--------------------
'''


class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.f1 = nn.ReLU()
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.f2 = nn.ReLU()
    self.l3 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    x = self.l1(x)
    x = self.f1(x)
    x = self.l2(x)
    x = self.f2(x)
    output = self.l3(x)

    return output
