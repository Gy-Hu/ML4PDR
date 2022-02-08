import z3
import numpy as np
import pandas as pd
import re
import pickle
from natsort import natsorted
import os

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
        self.db_gt = pd.read_csv(filename[1]) #ground truth of the label of literals (database) -> #TODO: refine here, only get one line for one object
        self.value_table = pd.read_pickle(filename[2])
        self.db_gt.drop("Unnamed: 0", axis=1, inplace=True)
        self.db_gt = self.db_gt.rename(columns={'nextcube': 'filename_nextcube'})
        self.db_gt = self.db_gt.reindex(natsorted(self.db_gt.columns), axis=1)
        self.n_vars = self.unpack_matrix.shape[1] - 1 #includes m and variable
        self.n_nodes = self.n_vars - (self.db_gt.shape[1] - 1) #only includes m
        index2list = self.check(str(filename[0]))
        self.is_flexible = (self.db_gt.values.tolist()[index2list])[1:] #TODO: refine here to locate automatically
        self.is_flexible = [int(x) for x in self.is_flexible]
        self.adj_matrix = self.unpack_matrix.copy()
        self.adj_matrix = self.adj_matrix.T.reset_index(drop=True).T
        self.adj_matrix.drop(self.adj_matrix.columns[0], axis=1, inplace=True)
        with open(filename[3], 'rb') as f:
            self.edges, self.relations, self.node_ref = pickle.load(f)
        # with open("../dataset/graph/" + filename[2], 'w') as p:
        #     g = pickle.load(p)


    def dump(self, dir, filename):
        dataset_filename = dir + (filename.split('/')[-1]).replace('adj_','')
        with open(dataset_filename, 'wb') as f_dump:
            pickle.dump(self, f_dump)

    def check(self, val):
        a = self.db_gt.index[self.db_gt['filename_nextcube'].str.contains(((val.split('/')[-1]).replace('.pkl', '.smt2')).replace('adj_',''),regex=False)]
        if a.empty:
            return 'not found'
        elif len(a) > 1:
            return a.tolist()
        else:
            # only one value - return scalar
            return a.item()

def mk_adj_matrix(filename):
    #filename = "../dataset/generalize_pre/nusmv.syncarb5^2.B_0.smt2"
    new_graph = generate_graph(filename)
    while len(new_graph.bfs_queue) != 0:
        new_graph.add()
    new_graph.print()
    new_graph.to_matrix()
    # with open("../dataset/graph/"+(filename.split('/')[-1]).replace('.smt2', '.pkl'), 'w') as p:
    #     pickle.dump(new_graph, p)
    new_graph.adj_matrix.to_pickle("../dataset/tmp/generalize_adj_matrix/"+"adj_"+(filename.split('/')[-1]).replace('.smt2', '.pkl'))
    new_graph.all_node_vt.to_pickle("../dataset/tmp/all_node_value_table/"+"vt_"+(filename.split('/')[-1]).replace('.smt2', '.pkl'))
    node_ref = {}
    for key,value in new_graph.node2nid.items():
        node_ref[value] = key.sexpr()
    with open("../dataset/tmp/edges_and_relation/"+"er_"+(filename.split('/')[-1]).replace('.smt2', '.pkl'), 'wb') as f:
        pickle.dump((new_graph.edges, new_graph.relations, node_ref), f)

def walkFile(dir):
    for root, _, files in os.walk(dir):
        files = natsorted(files)
        files = [os.path.join(root,f) for f in files]
    return files
        # for f in files:
        #     print(f)

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
    smt2_file_list = walkFile("../dataset/generalize_pre/")

    for smt2_file in smt2_file_list[200:]:
        mk_adj_matrix(smt2_file) # dump pkl with the adj_matrix -> should be refined later in problem class

    #FIXME: here still incomplete
    #refine_GT() # refine the ground truth by MUST tool

    adj_matrix_pkl_list = walkFile("../dataset/tmp/generalize_adj_matrix/")
    GT_table_csv_list = walkFile("../dataset/generalization/")
    vt_all_node_pkl_list = walkFile("../dataset/tmp/all_node_value_table/")
    edge_and_relation_pkl_list = walkFile("../dataset/tmp/edges_and_relation/")

    #TODO: Optimize the procedure of generate graph, for some smt2 file the graph generation is time consuming

    zipped = list(zip(adj_matrix_pkl_list, vt_all_node_pkl_list, edge_and_relation_pkl_list))
    raw_str_lst = []
    for raw in GT_table_csv_list:
        raw_str = (raw.split('/')[-1]).replace(".csv","")
        if any(raw_str in s for s in adj_matrix_pkl_list):
            raw_str_lst.append(raw_str)
    matching = []
    zipped_lst = list(map(lambda x: list(x), zipped))
    for substr in raw_str_lst:
        for item in zipped_lst:
            if any(substr in str_ for str_ in item):
                item.insert(1 , "../dataset/generalization/" + substr + ".csv")

    matching = [s for s in zipped_lst if len(s)!=3]
    # filename4prb = ["../dataset/tmp/generalize_adj_matrix/adj_nusmv.syncarb5^2.B_0.pkl","../dataset/generalization/nusmv.syncarb5^2.B.csv","../dataset/tmp/all_node_value_table/vt_nusmv.syncarb5^2.B_0.pkl","../dataset/tmp/edges_and_relation/er_nusmv.syncarb5^2.B_0.pkl"]
    for filename4prb in matching:
        prob = problem(filename4prb)
        prob.dump("../dataset/train/", filename4prb[0])
    #print(new_graph.adj_matrix)
