import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.l3 = nn.Linear(hidden_dim, out_dim)
    self.relu = nn.ReLU()
    #TODO: Modify to non-linear version for var_state

  def forward(self, x):
    x = self.l1(x)
    x = self.l2(x)
    x = self.l3(x)
    output = self.relu(x)

    return output

