# Try on deep walk and some other method to do graph embedding (semi-supervided, encode by embedding)
#TODO: Using deep walk and some methods to do embedding
#TODO: Using DAG-GNN to do embedding

# Try on MLP, GCN and transformer on undirected graph (encode by myself)
#TODO: Using AST, Bag-of-word, skip gram to encode the formula
#TODO: Using opcode (various encoding method for various opcode) to encode the formula
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import *
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GCNConv
import torch
from torch.nn import Linear
import torch.nn.functional as F
import z3
import deepwalk

class generate_graph:
    def __init__(self, filename):
        self.filename = filename
        self.constraints = z3.parse_smt2_file(filename, sorts={}, decls={})
        self.nid = 0
        self.node2nid = {}  # map z3expr -> node id
        self.relations = set()  # the edges
        self.edges = set()
        self.bfs_queue = [self.constraints[-1]]

    def getnid(self, node):
        if node in self.node2nid:
            nnid = self.node2nid[node]
        else:
            nnid = self.nid
            self.node2nid[node] = nnid
            self.nid += 1
        return nnid

    def add(self):
        n = self.bfs_queue[0]
        del self.bfs_queue[0]
        children = n.children()
        self.bfs_queue += list(children)

        nnid = self.getnid(n)
        op = n.decl().kind()
        if op == z3.Z3_OP_AND:
            opstr = 'AND'
        elif op == z3.Z3_OP_NOT:
            opstr = 'NOT'
        else:
            opstr = 'OTHER'

        if len(children) != 0:
            rel = f'{nnid} := {opstr} ( {[self.getnid(c) for c in children]} )'
            self.relations.add(rel)

        for c in children:
            cnid = self.getnid(c)
            self.edges.add((nnid, cnid))

    def print(self):
        print('-------------------')
        print('NODE:')
        for n, nid in self.node2nid.items():
            print(nid, ':', n)

        print('-------------------')
        print('RELATION:')
        relations = sorted(list(self.relations))
        for rel in relations:
            print(rel)

        print('-------------------')
        print('EDGE:')
        print(self.edges)

    def to_pyg(self):
        edges = list(self.edges)
        edges = [list(edges[i]) for i in range(0, len(edges), 1)]
        edges = sorted(edges, key=lambda x: (x[0], x[1]), reverse=False)
        edge_index_ori = torch.tensor(edges, dtype=torch.long) # directed graph
        edge_index = edge_index_ori.t().contiguous() # still directed graph
        edge_index = to_undirected(edge_index)
        return edge_index

    def to_file(self, edgelist_file):
        with open(edgelist_file, mode='w') as f:
            edges = sorted(self.edges, key=lambda x: (x[0], x[1]), reverse=False)
            for item in edges:
                f.write(str(item[0])+" "+str(item[1])+"\n")
            f.close()

class GAT(torch.nn.Module):
    def __init__(self, dataset):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.name = "gat"
        self.hidden_channels = 16
        self.conv1 = GATConv(dataset.num_features, self.hidden_channels)
        self.conv2 = GATConv(self.hidden_channels, dataset.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Train and test
def train(model, data):
    model.train()
    model.optimizer.zero_grad()
    if model.name == "mlp" : out = model(data.x) #only for mlp
    else: out = model(data.x, data.edge_index)
    loss = model.criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    model.optimizer.step()
    return loss

def test(model, data):
    model.eval()
    if model.name == "mlp":
        out = model(data.x)  # only for mlp
    else:
        out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

def test_nn_func():
    dataset = Planetoid(root='../dataset/Cora2', name='Cora')
    data = dataset[0]
    model = GAT(dataset) # initialize a model of MLP
    print(model)
    #train
    for epoch in range(1, 201):
        loss = train(model,data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    #test
    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    # test the function of nn
    # test_nn_func()
    filename = "../dataset/generalize_pre/nusmv.syncarb5^2.B_0.smt2"
    edgelist = "../dataset/edgelist/nusmv.syncarb5^2.B_0.txt"
    new_graph = generate_graph(filename)
    while len(new_graph.bfs_queue) != 0:
        new_graph.add()
    new_graph.print()
    new_graph.to_file(edgelist)
    embedding_dict = deepwalk.run(edgelist)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    #edge_list = new_graph.to_pyg()