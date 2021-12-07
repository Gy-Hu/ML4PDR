from z3 import *

class BMC:
    def __init__(self, primary_inputs, literals, primes, init, trans, post, pv2next):
        '''
        :param primary_inputs:
        :param literals: Boolean Variables
        :param primes: The Post Condition Variable
        :param init: The initial State
        :param trans: Transition Function
        :param post: The Safety Property
        '''
        self.primary_inputs = primary_inputs
        self.init = init
        self.trans = trans
        self.literals = literals + primary_inputs
        self.items = self.primary_inputs + self.literals
        self.lMap = {str(l): l for l in self.items}
        self.post = post
        self.frames = list()
        self.primes = primes + [Bool(str(pi)+'\'') for pi in primary_inputs]
        self.primeMap = [(self.literals[i], self.primes[i]) for i in range(len(self.literals))]
        self.pv2next = pv2next
        self.initprime = substitute(self.init.cube(), self.primeMap)
        self.vardict = dict()

    def vardef(self, n:str):
        if n in self.vardict:
            return self.vardict[n]
        v = Bool(n)
        self.vardict[n] = v
        return v

    def setup(self):
        self.slv = Solver()
        initmap = [(self.literals[i], self.vardef(str(self.literals[i])+"_0")) for i in range(len(self.literals))]
        self.slv.add(substitute(self.init.cube(), initmap))
        self.cnt = 0
    
    def get_map(self, idx):
        curr_map = [(self.literals[i], self.vardef(str(self.literals[i])+"_"+str(idx))) for i in range(len(self.literals))]
        next_map = [(self.primes[i], self.vardef(str(self.literals[i])+"_"+str(idx+1))) for i in range(len(self.literals))]
        return curr_map + next_map
    
    def unroll(self):
        idx = self.cnt
        var_map = self.get_map(idx)
        self.slv.add( substitute(self.trans.cube(), var_map) )
        self.cnt += 1
    
    def add(self, constraint):
        idx = self.cnt
        var_map = self.get_map(idx)
        self.slv.add( substitute(constraint, var_map) )
    
    def check(self):
        return self.slv.check()
        
        
        
        
