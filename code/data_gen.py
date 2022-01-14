import z3
import numpy as np
import pandas as pd
import re
import pickle
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
        self.all_node_var = {}
        self.solver = z3.Solver()

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
        self.calculate_node_value(n, nnid)  # calculate all node value ---> map to true/false
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

    def calculate_var(self):
        '''
        :return: a dictionary of variable and its index
        '''
        var = {}
        for item, value in self.node2nid.items():
            if not(re.match('And', str(item)) or re.match('Not',str(item))):
                var[value] = item
        return var
    #TODO: Add node value table
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
        n_nodes_var = len(node) #TODO: refine here
        A = np.zeros((n_nodes_var, n_nodes))
        df_2 = pd.DataFrame(self.all_node_var,index=[0]).T
        for edge in edges:
            i = int(edge[0])
            j = int(edge[1])
            weight = edge[2] #TODO: noted down the index of variable and node
            A[i, j] = weight #TODO: Encode the initial assigned value of the literals
            #A[j, i] = weight #TODO: modify here, encode the direction in the matrix
        self.adj_matrix = A
        df = pd.DataFrame(self.adj_matrix)

        def map(x):
            ori = x
            for key, value in var_dict.items():
                if key == x:
                    return "n_"+str(value)
            return "m_"+str(ori)

        df.rename(index=map, columns=map, inplace=True)
        df_2.rename(index=map, inplace=True)
        df_2.columns = ['Value']
        df_2 = df_2.reindex(natsorted(df_2.index), axis=0)
        #a = df.index.to_series().str.rsplit('_').str[0].sort_values()
        #df = df.reindex(index=a.index)
        df = df.reindex(natsorted(df.index), axis=0)
        df = df.reindex(natsorted(df.columns), axis=1)
        df = df.reset_index()
        df = df.rename(columns={'index': 'old_index'})
        df = df[~df.old_index.str.contains("n_")] #TODO: Refine here!!! remember to change here when change the index name of variable
        self.adj_matrix = df
        self.all_node_vt = df_2
        #print(df)

    def calculate_node_value(self, node, node_id):
        '''
        :return: the node value -> true or false
        '''
        self.solver.reset()
        self.solver.add(self.constraints[:-1])
        self.solver.add(node)
        if self.solver.check() == z3.sat:
            self.all_node_var[node_id] = 1 #-->sat so assign 1 as true
        else:
            self.all_node_var[node_id] = 0 #--> unsat so assign 0 as false

#TODO: ask question "where to encode the true/false value assignment of the variable"

class problem:
    def __init__(self, filename): # This class store the graph generated from generalized predessor (with serveral batches)
        # Should contain targets, which used for calculating loss
        self.filename = filename
        self.unpack_matrix = pd.read_pickle(filename[0])
        self.db_gt = pd.read_csv(filename[1])
        self.value_table = pd.read_pickle(filename[2])
        self.db_gt.drop("Unnamed: 0", axis=1, inplace=True)
        self.db_gt = self.db_gt.rename(columns={'nextcube': 'filename_nextcube'})
        self.db_gt = self.db_gt.reindex(natsorted(self.db_gt.columns), axis=1)
        self.n_vars = self.unpack_matrix.shape[1] - 1 #includes m and variable
        self.n_nodes = self.n_vars - (self.db_gt.shape[1] - 1) #only includes m
        self.is_flexible = (self.db_gt.values.tolist()[0])[1:]
        self.is_flexible = [int(x) for x in self.is_flexible]
        self.adj_matrix = self.unpack_matrix.copy()
        self.adj_matrix = self.adj_matrix.T.reset_index(drop=True).T
        self.adj_matrix.drop(self.adj_matrix.columns[0], axis=1, inplace=True)
        # with open("../dataset/graph/" + filename[2], 'w') as p:
        #     g = pickle.load(p)


    def dump(self, dir):
        dataset_filename = dir + "test.pkl"
        with open(dataset_filename, 'wb') as f_dump:
            pickle.dump(self, f_dump)

def mk_adj_matrix():
    filename = "../dataset/generalize_pre/nusmv.syncarb5^2.B_0.smt2"
    new_graph = generate_graph(filename)
    while len(new_graph.bfs_queue) != 0:
        new_graph.add()
    new_graph.print()
    new_graph.to_matrix()
    # with open("../dataset/graph/"+(filename.split('/')[-1]).replace('.smt2', '.pkl'), 'w') as p:
    #     pickle.dump(new_graph, p)
    new_graph.adj_matrix.to_pickle("../dataset/generalize_adj_matrix/"+(filename.split('/')[-1]).replace('.smt2', '.pkl'))
    new_graph.all_node_vt.to_pickle("../dataset/all_node_value_table/"+"vt_"+(filename.split('/')[-1]).replace('.smt2', '.pkl'))

#TODO: Refine ground truth data with MUST tool
def refine_GT():
    filename = "../dataset/generalization/nusmv.syncarb5^2.B.csv"

#TODO: ask question "how to set up the batch size"
def dump(self, dir):
    dataset_filename = dir
    with open(dataset_filename, 'wb') as f_dump:
        pickle.dump(batches, f_dump)

#TODO: Generate validation file here
def generate_val():
    filename = "../dataset/generalization/nusmv.syncarb5^2.B.csv"

#TODO: Collect more data for training
#TODO: one time to generate all data -> generalized the file name with enumerate
if __name__ == '__main__':
    mk_adj_matrix() # dump pkl with the adj_matrix -> should be refined later in problem class
    #refine_GT() # refine the ground truth by MUST tool
    filename4prb = ["../dataset/generalize_adj_matrix/nusmv.syncarb5^2.B_0.pkl","../dataset/generalization/nusmv.syncarb5^2.B.csv","../dataset/all_node_value_table/vt_nusmv.syncarb5^2.B_0.pkl"]
    prob = problem(filename4prb)
    prob.dump("../dataset/train/")
    #print(new_graph.adj_matrix)