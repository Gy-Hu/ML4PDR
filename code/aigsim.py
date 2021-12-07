class AIG(object):
    @staticmethod
    def _extract(literaleq):
        # we require the input looks like v==val
        children = literaleq.children()
        assert(len(children) == 2)
        if str(children[0]) in ['True', 'False']:
            v = children[1]
            val = children[0]
        elif str(children[1]) in ['True', 'False']:
            v = children[0]
            val = children[1]
        else:
            assert(False)
        return v, val

    def __init__(self):
        self.nodes = dict()
        self.node_use = dict()
        self.node_to_expr = dict()
    
    def register_latch(self, latchlist):
        self.latch = {int(l.var) for l in latchlist}
        self.latch_next = {int(latch.var) : int(latch.next) for latch in latchlist}
        
    def register_input(self, inputlist):
        self.input = {int(inp) for inp in inputlist}
            
    def register_ands(self, andlist):
        # make two graphs: 
        maxNum = 0
        for andgate in andlist:
            lhs = int(andgate.lhs); rhs0 = int(andgate.rhs0); rhs1 = int(andgate.rhs1)
            self.nodes[lhs] = [rhs0, rhs1]
            assert (lhs & 1 == 0)
            if rhs0 & 1 != 0:
                rhs0 -= 1
            if rhs1 & 1 != 0:
                rhs1 -= 1
                
            self.node_use[rhs0] = self.node_use.get(rhs0,[]) + [lhs]
            self.node_use[rhs1] = self.node_use.get(rhs1,[]) + [lhs]
            maxNum = max([maxNum, lhs, rhs0, rhs1])
        self.maxNum = maxNum
    
    def register_output(self, out):
        new_aig_to_add = {}
        Q = [out]
        while len(Q) > 0:
            n = Q[0]
            del Q[0]
            if n & 1 == 1:
                n -= 1
            if n not in self.nodes:
                assert (n in self.input or n in self.latch)
                continue
            RHS = self.nodes[n]
            new_aig_to_add[n] = RHS # no need to copy
            Q += RHS

        maxNum = self.maxNum + 2
        assert (maxNum not in self.nodes)

        new_nodes = sorted(new_aig_to_add.keys())
        for n in new_nodes:
            new_lhs = n + maxNum
            rhs = new_aig_to_add[n]
            new_rhs = []
            for node_used in rhs:
                if node_used & 1 == 1:
                    node_used -= 1
                    neg = 1
                else:
                    neg = 0

                if node_used in self.latch:
                    new_node_used = self.latch_next[node_used] ^ neg
                else:
                    new_node_used = node_used+maxNum+neg
                new_rhs.append(new_node_used)
                self.node_use[new_node_used^neg] = self.node_use.get(new_node_used^neg, []) + [new_lhs]
            self.nodes[new_lhs] = new_rhs

        self.input2next = {inp:inp+maxNum for inp in self.input}
        self.outnode = out + maxNum
        self.maxNum = maxNum*2 + 10
   
    def use_output(self):
        self.output = self.outnode
    def unuse_output(self):
        del self.output
        
    def register_c(self, c): # c should not contain input because we removed it already
        self.output = self.maxNum + 2
        self.reg_c = c
        clist = []
        for literal in c.cubeLiterals:
            var, val = AIG._extract(literal)
            nv = int(str(var)[1:])
            assert (nv in self.latch)
            nv_nxt = self.latch_next[nv]
            self.node_use[nv_nxt] = self.node_use.get(nv_nxt,[]) + [self.output]
            if val == True:
                clist.append(nv_nxt)
            else: # negating nv
                clist.append(nv_nxt ^ 1)
        self.nodes[self.output] = clist
    
    def unregister_c(self):
        c = self.reg_c
        for literal in c.cubeLiterals:
            var, val = AIG._extract(literal)
            nv = int(str(var)[1:])
            assert (nv in self.latch)
            nv_nxt = self.latch_next[nv]
            assert ( self.node_use[nv_nxt][-1] == self.output)
            del self.node_use[nv_nxt][-1]
        del self.nodes[self.output]
        del self.output
        del self.reg_c
    
    def get_output(self):
        return self.nodeval[self.output]
    
    def _compute(self, a):
        RHS = self.nodes[a]
        rhsvals = []
        # cal output
        for r in RHS:
            if r & 1 == 0:
                rhsvals.append(self.nodeval[r])
            else:
                rhsvals.append(1-self.nodeval[r-1])
            if rhsvals[-1] == -1:
                rhsvals[-1] = 2 # 1-2(X) --> -1
                
        res = rhsvals[0]
        for idx in range(1, len(rhsvals)):
            v = rhsvals[idx]
            # res = AND(res, v)
            if res == 0 or v == 0:
                res = 0
                break # that's 0
            elif res == 1: # 1 & 1  ,  1&X
                res = v
            elif res == 2: # X & 1,  X&X
                res = 2
            else:
                assert False
        return res

    def backtrack(self):
        def _getnode(idx):
            neg = idx&1
            idx = idx ^ neg
            return min(self.nodeval[idx]^neg,2)

        Q = [self.output]
        print ()
        while len(Q) > 0:
            n = Q[0]
            del Q[0]
            neg = n&1
            n = n^neg
            val = self.nodeval[n]
            print (n, ':', val, '<<--', self.nodes[n], [_getnode(idx) for idx in self.nodes[n]])
            if val == 2:
                Q += self.nodes[n]
            input()


    def populate_assignment_with_inputprime(self, model):
        self.nodeval = {0:0, 1:1}
        
        inputprime = {int(str(l)[1:-1]) : str(model[l]) for l in model if str(l)[-1] == '\'' and str(l)[0]=='i'}
        # input (no prime) -> True/False
        vassigned = set()
        for inp, TorF in inputprime.items():
            inp_prime = self.input2next[inp]
            boolv = 1 if TorF == 'True' else 0
            self.nodeval[inp_prime] = boolv
            vassigned.add(inp)
        unassigned_input = self.input.difference(vassigned)
        for inp in unassigned_input:
            self.nodeval[self.input2next[inp]] = 2
            
        noprime = {int(str(l)[1:]) : str(model[l]) for l in model if str(l)[-1] != '\''}
        
        vassigned = set()
        for nv,val in noprime.items():
            boolv = 0 if val == 'False' else 1
            self.nodeval[nv] = boolv
            vassigned.add(nv)
        
        unassgined_latch = self.latch.difference(vassigned)
        for l in unassgined_latch:
            self.nodeval[l] = 2
            
        unassigned_input = self.input.difference(vassigned)
        for l in unassigned_input:
            self.nodeval[l] = 2
            
        andLHS = sorted(self.nodes.keys())
        for a in andLHS:
            res = self._compute(a)
            self.nodeval[a] = res
            
    def populate_assignment(self, c):
        self.nodeval = {0:0, 1:1}
        vassigned = set()
        for literal in c.cubeLiterals:
            var, val = AIG._extract(literal)
            nv = int(str(var)[1:])
            boolv = 0 if str(val) == 'False' else 1
            self.nodeval[nv] = boolv
            vassigned.add(nv)
        
        unassgined_latch = self.latch.difference(vassigned)
        for l in unassgined_latch:
            self.nodeval[l] = 2
            
        unassigned_input = self.input.difference(vassigned)
        for l in unassigned_input:
            self.nodeval[l] = 2
            
        andLHS = sorted(self.nodes.keys())
        for a in andLHS:
            res = self._compute(a)
            self.nodeval[a] = res
                            
    def dump_eval(self):
        def _getnode(idx):
            neg = idx&1
            idx = idx ^ neg
            return min(self.nodeval[idx]^neg,2)
        ret = ""
        allnodes = sorted(self.nodes)
        for n in allnodes:
            ret += f"{n} {str(self.nodes[n])}" + str([_getnode(idx) for idx in self.nodes[n]]) + " -> " +\
                   str(_getnode(n)) + " used by " + str(self.node_use.get(n, [])) + \
                   " \n"
        return ret

    def set_Li(self, var, value, debug=False):
        #valueMap = {'True':1,'False':0,'X':2}
        nv = int(str(var)[1:])
        boolv = value
        self.nodeval[nv] = boolv
        Q = [nv]
        while len(Q) != 0:
          nv = Q[0]
          del Q[0]
          if nv not in self.node_use:
              continue
          to_eval = self.node_use[nv]
          for n in to_eval:
            assert (n&1 == 0)
            old_value = self.nodeval[n]
            res = self._compute(n)
            self.nodeval[n] = res
            if res != old_value:
                Q.append(n)
            
        
        
        # traverse the graph according to node_use, if it does not change then stop
        
        
        
        
