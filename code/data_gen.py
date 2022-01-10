import z3
import numpy as np
import pandas as pd
import re
from natsort import natsorted

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

    def to_matrix(self):
        edges = list(self.edges)
        edges = [list(edges[i]) for i in range(0, len(edges), 1)]
        edges = sorted(edges, key=lambda x: (x[0], x[1]), reverse=False)
        node = set()
        var_dict = self.calculate_var()
        for item in edges:
            node.add(item[0])
            node.add(item[1])
            item.append(1)
        n_nodes = len(node)
        n_nodes_var = len(node)
        A = np.zeros((n_nodes_var, n_nodes))
        for edge in edges:
            i = int(edge[0])
            j = int(edge[1])
            weight = edge[2]
            A[i, j] = weight #TODO: Encode the initial assigned value of the literals
            #A[j, i] = weight #TODO: modify here, encode the direction in the matrix
        self.adj_matrix = A
        df = pd.DataFrame(self.adj_matrix)

        def map(x):
            ori = x
            for key, value in var_dict.items():
                if key == x:
                    return "v_"+str(value)
            return "m_"+str(ori)

        df.rename(index=map, columns=map, inplace=True)
        #a = df.index.to_series().str.rsplit('_').str[0].sort_values()
        #df = df.reindex(index=a.index)
        df = df.reindex(natsorted(df.index), axis=0)
        df = df.reindex(natsorted(df.columns), axis=1)
        df = df.reset_index()
        df = df.rename(columns={'index': 'old_index'})
        df = df[~df.old_index.str.contains("v_")]
        self.adj_matrix = df
        #print(df)

    def calculate_var(self):
        var = {}
        for item, value in self.node2nid.items():
            if not(re.match('And', str(item)) or re.match('Not',str(item))):
                var[value] = item
        return var


# class problem:
#     def __init__(self): # This class store the graph generated from generalized predessor (with serveral batches)
#         # Should contain targets, which used for calculating loss

if __name__ == '__main__':
    filename = "../dataset/generalize_pre/nusmv.syncarb5^2.B_0.smt2"
    new_graph = generate_graph(filename)
    while len(new_graph.bfs_queue) != 0:
        new_graph.add()
    new_graph.print()
    new_graph.to_matrix()
    print(new_graph.adj_matrix)