'''
Parse the json file from ic3ref and build graph
'''
import argparse
import json
import sys
sys.path.append("..")
import model
import pdr
import matplotlib.pyplot as plt
from z3 import *

if __name__ == '__main__':
    #sys.stdout = open('file', 'w') #open this when we need the log
    help_info = "Usage: python main.py <file-name>.aag"
    parser = argparse.ArgumentParser(description="Run tests examples on the PDR algorithm")
    parser.add_argument('fileName', type=str, help='The name of the test to run', default=None, nargs='?')

    args = parser.parse_args(['../dataset/aag4train/eijk.S208o.S.aag']) 
    if (args.fileName is not None):
        file = args.fileName
        m = model.Model()
        solver = pdr.PDR(*m.parse(file))
    
    #with open('../dataset/json_graph/try_store_latch.json', 'r') as f: 
    with open('IC3ref/try_store_latch.json', 'r') as f: 
        latches = json.load(f)
    
    latches_lst = []
    latches_prime_lst = []

    #variable of latches
    vs = dict()
    s_l = Solver()
    for it in latches['latch']:
        name = "v" + str(it)
        vs[it] = Bool(name)
        #assign true to this variable
        latches_lst.append(vs[it])
    for i in range(len(latches_lst)):
        s_l.add(latches_lst[i]==True)

    # vars' of latch
    pvs = dict()
    s_lp = Solver()
    for it in latches['latch']:
        pvs[it] = Bool(str(it) + '_prime') # v -> v_prime, change this, because we want generate .smt2 later
        latches_prime_lst.append(pvs[it])
    for i in range(len(latches_prime_lst)):
        s_lp.add(latches_prime_lst[i]==True)

    s = pdr.tCube(0)
    for latch in s_l.assertions():
        s.add(latch)

    CTI = simplify(And(s.cubeLiterals))
    CTI_prime = substitute(substitute(substitute(simplify(And(s.cubeLiterals)), solver.primeMap),solver.inp_map),
                list(solver.pv2next.items()))

    pre_graph = And(Not(CTI),CTI_prime)
    
    
    print(latches_lst)
    print(latches_prime_lst)

