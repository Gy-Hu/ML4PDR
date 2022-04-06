from z3 import *
import time
import sys
import numpy as np
import copy
from queue import PriorityQueue
#from line_profiler import LineProfiler
from functools import wraps
import pandas as pd
from bmc import BMC
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from scipy.special import comb


# 查询接口中每行代码执行的时间
# def func_line_time(f):
#     @wraps(f)
#     def decorator(*args, **kwargs):
#         func_return = f(*args, **kwargs)
#         lp = LineProfiler()
#         lp_wrap = lp(f)
#         lp_wrap(*args, **kwargs)
#         lp.print_stats()
#         return func_return
#     return decorator

#TODO: Using Z3 to check the 3 properties of init, trans, safe, inductive invariant

# conjunction of literals.
class Frame:
    def __init__(self, lemmas):
        self.Lemma = lemmas
        self.pushed = [False] * len(lemmas)

    def cube(self):
        return And(self.Lemma)


    def add(self, clause, pushed=False):
        self.Lemma.append(clause)
        self.pushed.append(pushed)

    def __repr__(self):
        return str(sorted(self.Lemma, key=str))


class tCube:
    # make a tcube object assosciated with frame t.
    def __init__(self, t=0):
        self.t = t
        self.cubeLiterals = list()

    def __lt__(self, other):
        return self.t < other.t

    def clone(self):
        ret = tCube(self.t)
        ret.cubeLiterals = self.cubeLiterals.copy()
        return ret

    def remove_true(self):
        self.cubeLiterals = [c for c in self.cubeLiterals if c is not True]

#TODO: Using multiple timer to caculate which part of the code has the most time consumption
    # 解析 sat 求解出的 model, 并将其加入到当前 tCube 中 #TODO: lMap should incudes the v prime and i prime
    def addModel(self, lMap, model, remove_input): # not remove input' when add model
        no_var_primes = [l for l in model if str(l)[0] == 'i' or str(l)[-1] != '\''] # no_var_prime -> i2, i4, i6, i8, i2', i4', i6' or v2, v4, v6
        if remove_input:
            no_input = [l for l in no_var_primes if str(l)[0] != 'i'] # no_input -> v2, v4, v6
        else:
            no_input = no_var_primes # no_input -> i2, i4, i6, i8, i2', i4', i6' or v2, v4, v6
        # self.add(simplify(And([lMap[str(l)] == model[l] for l in no_input]))) # HZ:
        for l in no_input:
            self.add(lMap[str(l)] == model[l]) #TODO: Get model overhead is too high, using C API

    def remove_input(self):
        index_to_remove = set()
        for idx, literal in enumerate(self.cubeLiterals):
            children = literal.children()
            assert(len(children) == 2)

            if str(children[0]) in ['True', 'False']:
                v = str(children[1])
            elif str(children[1]) in ['True', 'False']:
                v = str(children[0])
            else:
                assert(False)
            assert (v[0] in ['i', 'v'])
            if v[0] == 'i':
                index_to_remove.add(idx)
        self.cubeLiterals = [self.cubeLiterals[i] for i in range(len(self.cubeLiterals)) if i not in index_to_remove]


    # 扩增 CNF 式
    def addAnds(self, ms):
        for i in ms:
            self.add(i)

    # 增加一个公式到当前 tCube() 中
    def add(self, m):
        self.cubeLiterals.append(m) # HZ: does not convert to cnf for the time being
        # g = Goal()
        # g.add(m) #TODO: Check 这边CNF会不会出现问题（试试arb-start那个case）
        # t = Tactic('tseitin-cnf')  # 转化得到该公式的 CNF 范式 #TODO:弄清楚这边转CNF如何转，能不能丢入Parafrost加速
        # for c in t(g)[0]:
        #     self.cubeLiterals.append(c)
        # if len(t(g)[0]) == 0:
        #     self.cubeLiterals.append(True)

    def true_size(self):
        '''
        Remove the 'True' in list (not the BoolRef Variable)
        '''
        return len(self.cubeLiterals) - self.cubeLiterals.count(True) 

    def join(self,  model):
        # first extract var,val from cubeLiteral
        literal_idx_to_remove = set()
        model = {str(var): model[var] for var in model}
        for idx, literal in enumerate(self.cubeLiterals):
            if literal is True:
                continue
            var, val = _extract(literal)
            var = str(var)
            assert(var[0] == 'v')
            if var not in model:
                literal_idx_to_remove.add(idx)
                continue
            val2 = model[var]
            if str(val2) == str(val):
                continue # will not remove
            literal_idx_to_remove.add(idx)
        for idx in literal_idx_to_remove:
            self.cubeLiterals[idx] = True
        return len(literal_idx_to_remove) != 0
        # for each variable in cubeLiteral, check if it has negative literal in model
        # if so, remove this literal
        # return False if there is no removal (which should not happen)


    # 删除第 i 个元素，并返回新的tCube
    def delete(self, i: int):
        res = tCube(self.t)
        for it, v in enumerate(self.cubeLiterals):
            if i == it:
                res.add(True)
                continue
            res.add(v)
        return res

    #TODO: 验证这个cube()是否导致了求解速度变慢

    def cube(self): #导致速度变慢的罪魁祸首？
        return simplify(And(self.cubeLiterals))

    # Convert the trans into real cube
    def cube_remove_equal(self):
        res = tCube(self.t)
        for literal in self.cubeLiterals:
            children = literal.children()
            assert(len(children) == 2)
            cube_literal = And(Not(And(children[0],Not(children[1]))), Not(And(children[1],Not(children[0]))))
            res.add(cube_literal)
        return res


    # def ternary_sim(self, index_of_x):
    #     # first extract var,val from cubeLiteral
    #     s = Solver()
    #     for idx, literal in enumerate(self.cubeLiterals):
    #         if idx !=index_of_x:
    #             s
    #             var = str(var)
    #
    # def cube(self):
    #     return And(*self.cubeLiterals)

    def __repr__(self):
        return str(self.t) + ": " + str(sorted(self.cubeLiterals, key=str))

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

class PDR:
    def __init__(self, primary_inputs, literals, primes, init, trans, post, pv2next, primes_inp, filename, smt2_gen_GP=0, smt2_gen_IG=0):
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
        self.literals = literals
        self.items = self.primary_inputs + self.literals + primes_inp + primes
        self.lMap = {str(l): l for l in self.items}
        self.post = post 
        self.frames = list()
       # self.primaMap_new = [(literals[i], primes[i]) for i in range(len(literals))] #TODO: Map the input to input' (input prime)
        self.primeMap = [(literals[i], primes[i]) for i in range(len(literals))]
        self.inp_map = [(primary_inputs[i], primes_inp[i]) for i in range(len(primes_inp))]
        #self.inp_prime = primes_inp
        self.pv2next = pv2next
        self.initprime = substitute(self.init.cube(), self.primeMap)
        # for debugging purpose
        self.bmc = BMC(primary_inputs=primary_inputs, literals=literals, primes=primes,
                       init=init, trans=trans, post=post, pv2next=pv2next, primes_inp = primes_inp)
        self.generaliztion_data_GP = []# Store the ground truth data of generalized predecessor
        self.generaliztion_data_IG = []# Store the ground truth data of inductive generalization 
        #TODO: Use self.generaliztion_data_IG to store the ground truth data of inductive generalization 
        self.filename = filename
        '''
        --------------The following variables are used to calculate the reducing rate--------
        '''
        self.sum_MIC = 0 # Sum of the literals produced by MIC
        self.sum_IG_GT = 0 # Sum of the literals produced by combinations
        self.sum_GP = 0 # Sum of the literals of predecessor (unsat core or other methods)
        self.sum_GP_GT = 0 #Sum of the minimum literals of predecessor (MUST, ternary simulation etc.)
        '''
        --------------Switch to open/close the ground truth data generation------------------
        '''
        self.smt2_gen_IG = smt2_gen_IG
        self.smt2_gen_GP = smt2_gen_GP
        

    def check_init(self):
        s = Solver()
        s.add(self.init.cube())
        s.add(Not(self.post.cube()))
        res1 = s.check()
        if res1 == sat:
            return False
        s = Solver()
        s.add(self.init.cube())
        s.add(self.trans.cube())
        s.add(substitute(substitute(Not(self.post.cube()), self.primeMap),self.inp_map))
        res2 = s.check()
        if res2 == sat:
            return False
        return True

    def run(self, agent=None):

        if not self.check_init():
            print("Found trace ending in bad state")
            return False

        self.agent = agent
        self.frames = list() # list for Frame
        self.frames.append(Frame(lemmas=[self.init.cube()]))
        self.frames.append(Frame(lemmas=[self.post.cube()]))

        while True:
            c = self.getBadCube() # conduct generalize predecessor here
            if c is not None:
                # print("get bad cube!")
                trace = self.recBlockCube(c) # conduct generalize predecessor here (in solve relative process)
                #TODO: 找出spec3-and-env这个case为什么没有recBlock
                if trace is not None:
                    # Generate ground truth of generalized predecessor
                    if self.generaliztion_data_GP: # When this list is not empty, it return true
                        df = pd.DataFrame(self.generaliztion_data_GP)
                        df = df.fillna(1)
                        df.to_csv("../dataset/GP2graph/generalization/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                    
                    # Generate ground truth of inductive generalization
                    if self.generaliztion_data_IG: # When this list is not empty, it return true
                        df = pd.DataFrame(self.generaliztion_data_IG)
                        df = df.fillna(0)
                        df.to_csv("../dataset/IG2graph/generalization/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                    
                    # Print out the improvement space of inductive generalization
                    if self.sum_IG_GT != 0:
                        print("Sum of the literals produced by MIC(): ",self.sum_MIC)
                        print("Sum of the literals produced by enumeration in indutive generalization: ",self.sum_IG_GT)
                        print("Reducing ",((self.sum_MIC-self.sum_IG_GT)/self.sum_MIC)*100,"% ")
                    
                    print("Found trace ending in bad state:")
                    
                    
                    self._debug_trace(trace)
                    # If want to print the trace, remember to comment the _debug_trace() function
                    while not trace.empty():
                        idx, cube = trace.get()
                        print(cube)
                    return False
                print("recBlockCube Ok! F:")

            else:
                inv = self.checkForInduction()
                if inv != None:
                    print("Found inductive invariant")
                    # Print out the improvement space of inductive generalization
                    if self.sum_IG_GT != 0:
                        print("Sum of the literals produced by MIC(): ",self.sum_MIC)
                        print("Sum of the literals produced by enumeration in indutive generalization: ",self.sum_IG_GT)
                        print("Reducing ",((self.sum_MIC-self.sum_IG_GT)/self.sum_MIC)*100,"% ")
                    
                    
                    # Generate ground truth of generalized predecessor
                    if self.generaliztion_data_GP: # When this list is not empty, it return true
                        df = pd.DataFrame(self.generaliztion_data_GP)
                        df = df.fillna(1)
                        #df.iloc[:,2:] = df.iloc[:,2:].apply(pd.to_numeric)
                        df.to_csv("../dataset/GP2graph/generalization/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                    
                    # Generate ground truth of inductive generalization
                    if self.generaliztion_data_IG: # When this list is not empty, it return true
                        df = pd.DataFrame(self.generaliztion_data_IG)
                        df = df.fillna(0)
                        df.to_csv("../dataset/IG2graph/generalization/" + (self.filename.split('/')[-1]).replace('.aag', '.csv'))
                    
                    print ('Total F', len(self.frames), ' F[-1]:', len(self.frames[-1].Lemma))
                    self._debug_print_frame(len(self.frames)-1)

                    return True
                print("Did not find invariant, adding frame " + str(len(self.frames)) + "...")


                print("Adding frame " + str(len(self.frames)) + "...")
                self.frames.append(Frame(lemmas=[])) # property can be directly pushed here

                # TODO: Append P, and get bad cube change to F[-1] /\ T /\ !P' (also can do generalization), check it is sat or not
                # [init, P]
                # init /\ bad   ?sat
                # init /\T /\ bad'  ?sat

                #TODO: Try new way to pushing lemma (like picking >=2 clause at once to add in new frame)
                for idx in range(1,len(self.frames)-1):
                    self.pushLemma(idx)

                self._sanity_check_frame()
                print("Now print out the size of frames")
                for index in range(len(self.frames)):
                    push_cnt = self.frames[index].pushed.count(True)
                    print("F", index, 'size:', len(self.frames[index].Lemma), 'pushed: ', push_cnt)
                    assert (len(self.frames[index].Lemma) == len(self.frames[index].pushed))
                for index in range(1, len(self.frames)):
                    print (f'--------F {index}---------')
                    self._debug_print_frame(index, skip_pushed=True)
                #input() # pause





    def checkForInduction(self):
        #print("check for Induction now...")
        # check Fi+1 => Fi ?
        Fi2 = self.frames[-2].cube()
        Fi = self.frames[-1].cube()
        s = Solver()
        s.add(Fi)
        s.add(Not(Fi2))
        if s.check() == unsat:
            return Fi
        return None


    def pushLemma(self, Fidx:int):
        fi: Frame = self.frames[Fidx]
        for lidx, c in enumerate(fi.Lemma):
            if fi.pushed[lidx]:
                continue
            s = Solver()
            s.add(fi.cube())
            s.add(self.trans.cube())
            s.add(substitute(Not(substitute(c, self.primeMap)),self.inp_map))
            if s.check() == unsat:
                fi.pushed[lidx] = True
                self.frames[Fidx + 1].add(c)

    def frame_trivially_block(self, st: tCube):
        Fidx = st.t
        slv = Solver()
        slv.add(self.frames[Fidx].cube())
        slv.add(st.cube())
        if slv.check() == unsat:
            return True
        return False

    #TODO: 解决这边特殊case遇到safe判断成unsafe的问题
    def recBlockCube(self, s0: tCube):
        '''
        :param s0: CTI (counterexample to induction, represented as cube)
        :return: Trace (cex, indicates that the system is unsafe) or None (successfully blocked)
        '''
        Q = PriorityQueue()
        print("recBlockCube now...")
        Q.put((s0.t, s0))
        prevFidx = None
        while not Q.empty():
            print (Q.qsize())
            s:tCube = Q.get()[1]
            if s.t == 0:
                return Q

            assert(prevFidx != 0)
            if prevFidx is not None and prevFidx == s.t-1:
                # local lemma push
                self.pushLemma(prevFidx)
            prevFidx = s.t
            # check Frame trivially block
            if self.frame_trivially_block(s):
                #Fmin = s.t+1
                #Fmax = len(self.frames)
                #if Fmin < Fmax:
                #    s_copy = s.clone()
                #    s_copy.t = Fmin
                #    Q.put((Fmin, s_copy)) #TODO: Open this will cause the problem in bmc check
                continue

            z = self.solveRelative(s)
            if z is None:
                sz = s.true_size()
                original_s = s.clone()
                s_enumerate = self.generate_GT(original_s) #Generate ground truth here
                s = self.MIC(s)
                print ('MIC ', sz, ' --> ', s.true_size(),  'F', s.t)
                self.sum_MIC = self.sum_MIC + s.true_size()
                if s_enumerate is not None: 
                    print ("Find minimum", sz,' --> ', s_enumerate.true_size(),  'F', s_enumerate.t)
                    self.sum_IG_GT = self.sum_IG_GT + s_enumerate.true_size()
                else:
                    print ("Minimum not found")
                    self.sum_IG_GT = self.sum_IG_GT + s.true_size()
                self._check_MIC(s)
                self.frames[s.t].add(Not(s.cube()), pushed=False)
                for i in range(1, s.t):
                    self.frames[i].add(Not(s.cube()), pushed=True) #TODO: Try RL here
                # reQueue : see IC3 PDR Friends
                #Fmin = original_s.t+1
                #Fmax = len(self.frames)
                #if Fmin < Fmax:
                #    s_copy = original_s #s.clone()
                #    s_copy.t = Fmin
                #    Q.put((Fmin, s_copy))

            else: #SAT condition
                assert(z.t == s.t-1)
                Q.put((s.t, s))
                Q.put((s.t-1, z))
        return None


    def recBlockCube_RL(self, s0: tCube):
        print("recBlockCube now...")
        Q = [s0]
        while len(Q) > 0:
            s = Q[-1]
            if s.t == 0:
                return Q

            # solve if cube s was blocked by the image of the frame before it
            z, u = self.solveRelative_RL(s)

            if (z == None):
                # Cube 's' was blocked by image of predecessor:
                # block cube in all previous frames
                Q.pop()  # remove cube s from Q
                for i in range(1, s.t + 1):
                    # if not self.isBlocked(s, i):
                    self.R[i] = And(self.R[i], Not(u))
            else:
                # Cube 's' was not blocked by image of predecessor
                # it will stay on the stack, and z (the model which allowed transition to s) will we added on top
                Q.append(z)
        return None

    def _solveRelative(self, tcube) -> tCube:
        '''
        #FIXME: The inductive relative checking should subtitue input -> input'
        '''
        #cubePrime = substitute(tcube.cube(), self.primeMap)
        cubePrime = substitute(substitute(tcube.cube(), self.primeMap),self.inp_map)
        s = Solver()
        s.add(Not(tcube.cube()))
        s.add(self.frames[tcube.t - 1].cube())
        s.add(self.trans.cube())
        s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
        return s.check()

    def _test_MIC1(self, q: tCube):
        passed_single_q = []
        for i in range(len(q.cubeLiterals)):
            qnew = tCube(q.t)
            var, val = _extract(q.cubeLiterals[i])
            if str(val) != "True": # will intersect with init: THIS IS DIRTY check
                continue
            qnew.cubeLiterals = [q.cubeLiterals[i]]
            if self._solveRelative(qnew) == unsat:
                passed_single_q.append(qnew)
        return passed_single_q

    def _test_MIC2(self, q: tCube):
        def check_init(c: tCube):
            slv = Solver()
            slv.add(self.init.cube())
            slv.add(c.cube())
            return slv.check()

        passed_single_q = []
        for i in range(len(q.cubeLiterals)):
            for j in range(i+1, len(q.cubeLiterals)):
                qnew = tCube(q.t)
                qnew.cubeLiterals = [q.cubeLiterals[i], q.cubeLiterals[j]]
                if check_init(qnew) == sat:
                    continue
                if self._solveRelative(qnew) == unsat:
                    passed_single_q.append(qnew)
        return passed_single_q
        

    def MIC(self, q: tCube): #TODO: Check the algorithm is correct or not
        #passed_single_q_sz1 = self._test_MIC1(q)
        #passed_single_q_sz2 = []
        #if len(passed_single_q_sz1) == 0:
        #    passed_single_q_sz2 = self._test_MIC2(q)

        sz = q.true_size()
        self.unsatcore_reduce(q, trans=self.trans.cube(), frame=self.frames[q.t-1].cube())
        print('unsatcore', sz, ' --> ', q.true_size())
        q.remove_true()

        for i in range(len(q.cubeLiterals)):
            if q.cubeLiterals[i] is True:
                continue
            q1 = q.delete(i)
            print(f'MIC try idx:{i}')
            if self.down(q1): 
                q = q1
        q.remove_true()
        print (q)
        # FIXME: below shows the choice of var is rather important
        # I think you may want to first run some experience to confirm
        # that if can achieve minimum, it will be rather useful
        # if q.true_size() > 1 and len(passed_single_q_sz1) != 0:
        #     q = passed_single_q_sz1[0] # should be changed!
        #     print ('Not optimal!!!')
        # if q.true_size() > 2 and len(passed_single_q_sz2) != 0:
        #     for newq in passed_single_q_sz2:
        #         if 'False' in str(newq): #Ask this, why is 'False' in str(newq)?
        #             q = newq
        #     # should be changed!
        #     print ('Not optimal!!!')
        return q
        # i = 0
        # while True:
        #     print(i)
        #     if i < len(q.cubeLiterals) - 1:
        #         i += 1
        #     else:
        #         break
        #     q1 = q.delete(i)
        #     if self.down(q1):
        #         q = q1
        # return q
    
    #TODO: Add assertion on this to check inductive relative
    #TODO: Add assertion to check there is no 'True' and 'False' in the cubeLiterals list
    def generate_GT(self,q: tCube): #smt2_gen_IG is a switch to trun on/off .smt file generation
        
        if self.smt2_gen_IG == 0:
            return None
        elif self.smt2_gen_IG == 1:
            assert(q.cubeLiterals.count(True)==0)
            assert(q.cubeLiterals.count(False)==0)
            '''
            ---------------------Generate .smt2 file (for building graph)--------------
            
            
            #FIXME: This .smt generation still exists problem, remember to fix this
            s_smt = Solver()  #use to generate SMT-lib2 file

            #This "Cube" is a speical circuit of combining two conditions of solve relative (determine inductive generalization)

            # s_smt.add(Not(q.cube()))
            # s_smt.add(self.frames[q.t - 1].cube())
            # s_smt.add(self.trans.cube())
            # s_smt.add(substitute(substitute(q.cube(), self.primeMap),self.inp_map))
            
            Cube = Not(
                And(
                    Not(
                      And(self.frames[q.t-1].cube(), Not(q.cube()), self.trans.cube_remove_equal().cube(),
                      substitute(substitute(substitute(q.cube(), self.primeMap),self.inp_map),list(self.pv2next.items())))
                      #substitute(q.cube(), self.primeMap))
                      )  # Fi-1 ! and not(q) and T and q'
                ,
                    Not(And(self.frames[0].cube(),q.cube()))
                    )
            )

            #Cube = substitute(Cube,list(self.pv2next.items()))

            for index, literals in enumerate(q.cubeLiterals): 
                s_smt.add(literals) 
                # s_smt.assert_and_track(literals,'p'+str(index))
            
            s_smt.add(Cube)  # F[i - 1] and T and Not(badCube) and badCube'

            assert (s_smt.check() == unsat)

            filename = '../dataset/IG2graph/generalize_IG/' + (self.filename.split('/')[-1]).replace('.aag', '_'+ str(len(self.generaliztion_data_IG)) +'.smt2')
            with open(filename, mode='w') as f:
                f.write(s_smt.to_smt2())
            f.close()
            '''
            

            '''
            -------------------Generate ground truth--------------
            '''

            def check_init(c: tCube):
                slv = Solver()
                slv.add(self.init.cube())
                slv.add(c.cube())
                return slv.check()

            # sz = q.true_size()
            # self.unsatcore_reduce(q, trans=self.trans.cube(), frame=self.frames[q.t-1].cube())
            # print('unsatcore', sz, ' --> ', q.true_size())
            # q.remove_true()

            end_lst = []
            passed_minimum_q = []
            is_looping = True
            for i in range(1,len(q.cubeLiterals)+1): #When i==len(q.cubeLiterals), this means it met wrost case
                for c in combinations(q.cubeLiterals, i):
                    if len(end_lst) > 3000:
                        is_looping = False
                        break
                    end_lst.append(c)
                if is_looping==False:
                    break

            #FIXME: This may cause memory exploration of list (length 2^n, n is the length of original q)

            '''
            1 -> 0
            2 -> Cn1
            3 -> Cn1+Cn2
            4 -> Cn1+Cn2+Cn3
            5 -> Cn1+Cn2+Cn3+Cn4
            ...
            n -> Cn1+Cn2+Cn3+Cn4+...+Cnn -> 2^n - 1 
            '''
            
            #TODO: Using multi-thread to handle inductive relative checking
            # dict_n = {}
            # dict_n[1] = 0
            # dict_n[2] = int(comb(len(end_lst),1))
            # dict_n[3] = int(comb(len(end_lst),1) + comb(len(end_lst),2))
            # dict_n[4] = int(comb(len(end_lst),1) + comb(len(end_lst),2) \
            #     + comb(len(end_lst),2)+comb(len(end_lst),3))
            
            data = {} # Store ground truth, and output to .csv
            for tuble in end_lst:
                if len(passed_minimum_q) > 0:
                    break
                elif len(passed_minimum_q) == 0:
                    qnew = tCube(q.t)
                    qnew.cubeLiterals = [tcube for tcube in tuble]
                    if check_init(qnew) == sat:
                        continue
                    # if self._solveRelative(qnew) == sat:
                    #     print("Did not pass inductive relative check")
                    #     continue
                    if self._solveRelative(qnew) == unsat:
                        passed_minimum_q.append(qnew)
                else:
                    raise AssertionError
                #ADD: When len(passed_single_q) != 0, break the for loop
            if len(passed_minimum_q)!= 0:
                '''
                ---------------------Generate .smt2 file (for building graph)--------------

                Not generate the .smt2 file when enumerate combinations of literals could not find ground truth
                '''
                
                s_smt = Solver()
                Cube = Not(
                    And(
                        Not(
                        And(self.frames[q.t-1].cube(), 
                        Not(q.cube()), 
                        substitute(substitute(substitute(q.cube(), self.primeMap),self.inp_map),
                        list(self.pv2next.items()))
                        )),
                        Not(And(self.frames[0].cube(),q.cube()))
                        ))
                for index, literals in enumerate(q.cubeLiterals): s_smt.add(literals)
                s_smt.add(Cube)
                assert (s_smt.check() == unsat)
                filename = '../dataset/IG2graph/generalize_IG/' + (self.filename.split('/')[-1]).replace('.aag', '_'+ str(len(self.generaliztion_data_IG)) +'.smt2')
                data['inductive_check'] = filename.split('/')[-1] #Store the name of .smt file
                with open(filename, mode='w') as f: f.write(s_smt.to_smt2())
                f.close() 
                

                '''
                ---------------------Export the ground truth----------------------
                '''
                q_minimum = passed_minimum_q[0] # Minimum ground truth has been generated
                for idx in range(len(q.cubeLiterals)): # -> ground truth size is q
                    var, val = _extract(q.cubeLiterals[idx])
                    data[str(var)] = 0
                # for idx in range(len(Cube.cubeLiterals)): # -> ground truth size is Cube (combine of two check)
                #     var, val = _extract(Cube.cubeLiterals[idx])
                #     data[str(var)] = 0
                for idx in range(len(q_minimum.cubeLiterals)):
                    var, val = _extract(q_minimum.cubeLiterals[idx])
                    data[str(var)] = 1 # Mark q-like as 1
                self.generaliztion_data_IG.append(data)
                return q_minimum
            else:
                print("The ground truth has not been found")
                return None


    def unsatcore_reduce(self, q:  tCube, trans, frame):
        # (( not(q) /\ F /\ T ) \/ init' ) /\ q'   is unsat
        slv = Solver()
        slv.set(unsat_core=True)

        l = Or( And(Not(q.cube()), trans, frame), self.initprime)
        slv.add(l)

        plist = []
        for idx, literal in enumerate(q.cubeLiterals):
            p = 'p'+str(idx)
            slv.assert_and_track(substitute(substitute(literal, self.primeMap),self.inp_map), p)
            plist.append(p)
        res = slv.check()
        if res == sat:
            model = slv.model()
            print(model.eval(self.initprime))
            assert False
        assert (res == unsat)
        core = slv.unsat_core()
        for idx, p in enumerate(plist):
            if Bool(p) not in core:
                q.cubeLiterals[idx] = True
        return q


    def down(self, q: tCube):
        while True:
            print(q.true_size(), end=',')
            s = Solver()
            s.push()
            #s.add(And(self.frames[0].cube(), Not(q.cube())))
            s.add(self.frames[0].cube())
            s.add(q.cube())
            #if unsat == s.check():
            if sat == s.check():
                print('F')
                return False
            s.pop()
            s.push()
            s.add(And(self.frames[q.t-1].cube(), Not(q.cube()), self.trans.cube(), #TODO: Check here is t-1 or t
                      substitute(substitute(q.cube(), self.primeMap),self.inp_map)))  # Fi-1 ! and not(q) and T and q'
            if unsat == s.check():
                print('T')
                return True
            # TODO: this is not the down process !!!
            m = s.model()
            has_removed = q.join(m)
            s.pop()
            assert (has_removed)
            #return False

    # def tcgMIC(self, q: tCube, d: int):
    #     for i in range(len(q.cubeLiterals)):
    #         q1 = q.delete(i)
    #         if self.ctgDown(q1, d):
    #             q = q1
    #     return q
    #
    # def ctgDown(self, q: tCube, d: int):
    #     ctgs = 0
    #     while True:
    #         s = Solver()
    #         s.push()
    #         s.add(And(self.R[0].cube(), Not(q.cube())))
    #         if unsat == s.check():
    #             return False
    #         s.pop()
    #         s.push()
    #         s.add(And(self.R[q.t].cube(), Not(q.cube()), self.trans.cube(),
    #                   substitute(q.cube(), self.primeMap)))  # Fi and not(q) and T and q'
    #         if unsat == s.check():
    #             return True
    #         m = s.model()

    def _debug_print_frame(self, fidx, skip_pushed=False):
        for idx, c in enumerate(self.frames[fidx].Lemma):
            if skip_pushed and self.frames[fidx].pushed[idx]:
                continue
            if 'i' in str(c):
                print('C', idx, ':', 'property')
            else:
                print('C', idx, ':', str(c))


    def _debug_c_is_predecessor(self, c, t, f, not_cp):
        s = Solver()
        s.add(c)
        s.add(t)
        if f is not True:
            s.add(f)
        s.add(not_cp)
        assert (s.check() == unsat)

    # tcube is bad state

    def _check_MIC(self, st:tCube):
        cubePrime = substitute(substitute(st.cube(), self.primeMap),self.inp_map)
        s = Solver()
        s.add(Not(st.cube()))
        s.add(self.frames[st.t - 1].cube())
        s.add(self.trans.cube())
        s.add(cubePrime)
        assert (s.check() == unsat)

    # for tcube, check if cube is blocked by R[t-1] AND trans (check F[i−1]/\!s/\T/\s′ is sat or not)
    def solveRelative(self, tcube) -> tCube:
        '''
        :param tcube: CTI (counterexample to induction, represented as cube)
        :return: None (relative solved! Begin to block bad state) or
        predecessor to block (Begin to enter recblock() again)
        '''
        cubePrime = substitute(substitute(tcube.cube(), self.primeMap),self.inp_map)
        s = Solver()
        s.add(Not(tcube.cube()))
        s.add(self.frames[tcube.t - 1].cube())
        s.add(self.trans.cube())
        s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
        if s.check() == sat: # F[i-1] & !s & T & s' is sat!!
            model = s.model()
            c = tCube(tcube.t - 1)
            c.addModel(self.lMap, model, remove_input=False)  # c = sat_model, get the partial model of c
            #return c
            print("cube size: ", len(c.cubeLiterals), end='--->')
            # FIXME: check1 : c /\ T /\ F /\ Not(cubePrime) : unsat
            self._debug_c_is_predecessor(c.cube(), self.trans.cube(), self.frames[tcube.t-1].cube(), Not(cubePrime))
            generalized_p = self.generalize_predecessor(c, next_cube_expr = tcube.cube())  # c = get_predecessor(i-1, s')
            print(len(generalized_p.cubeLiterals))
            #
            # FIXME: sanity check: gp /\ T /\ F /\ Not(cubePrime)  unsat
            self._debug_c_is_predecessor(generalized_p.cube(), self.trans.cube(), self.frames[tcube.t-1].cube(), Not(cubePrime))
            generalized_p.remove_input()
            return generalized_p #TODO: Using z3 eval() to conduct tenary simulation
        return None

    #(X ∧ 0 = 0), (X ∧ 1 = X), (X ∧ X = X), (¬X = X).
    # def ternary_operation(self, ternary_candidate):
    #     for
    #     False = And(x,True)
    #     x = And(x,True)
    #     x = And(x,x)
    #     x = Not(x)

#TODO: Get bad cude should generalize as well!
    def generalize_predecessor(self, prev_cube:tCube, next_cube_expr): #smt2_gen_GP is a switch to trun on/off .smt file generation
        '''
        :param prev_cube: sat model of CTI (v1 == xx , v2 == xx , v3 == xxx ...)
        :param next_cube_expr: bad state (or CTI), like !P ( ? /\ ? /\ ? /\ ? .....)
        :return:
        '''
        data = {}
        #data['previous cube'] = str(prev_cube)
        #data['previous cube'] = (prev_cube.cube()).sexpr()


        #check = tcube.cube()

        tcube_cp = prev_cube.clone() #TODO: Solve the z3 exception warning
        ground_true = prev_cube.clone()
        #print("original size of !P (or CTI): ", len(tcube_cp.cubeLiterals))
        #print("Begin to generalize predessor")

        #replace the state as the next state (by trans) -> !P (s')
        nextcube = substitute(substitute(substitute(next_cube_expr, self.primeMap),self.inp_map), list(self.pv2next.items())) # s -> s'
        #data['nextcube'] = str(nextcube)

        # try:
        #     nextcube = substitute(substitute(next_cube_expr, self.primeMap), list(self.pv2next.items()))
        # except Exception:
        #     pass
        index_to_remove = []

        #sanity check
        #s = Solver()
        #s.add(prev_cube.cube())
        #s.check()
        #assert(str(s.model().eval(nextcube)) == 'True')
        
        if self.smt2_gen_GP==1:
            s = Solver()
            s_smt = Solver()  #use to generate SMT-lib2 file
            for index, literals in enumerate(tcube_cp.cubeLiterals):
                s_smt.add(literals) 
                s.assert_and_track(literals,'p'+str(index)) # -> ['p1','p2','p3']
            s.add(Not(nextcube))
            s_smt.add(Not(nextcube)) 
            assert(s.check() == unsat and s_smt.check() == unsat)
            core = s.unsat_core()
            core = [str(core[i]) for i in range(0, len(core), 1)] # -> ['p1','p3'], core -> nextcube

            filename = '../dataset/GP2graph/generalize_pre/' + (self.filename.split('/')[-1]).replace('.aag', '_'+ str(len(self.generaliztion_data_GP)) +'.smt2')
            data['nextcube'] = filename.split('/')[-1]
            with open(filename, mode='w') as f:
                f.write(s_smt.to_smt2())
            f.close()

        elif self.smt2_gen_GP==0:
            s = Solver()
            for index, literals in enumerate(tcube_cp.cubeLiterals):
                s.assert_and_track(literals,'p'+str(index)) # -> ['p1','p2','p3']
            s.add(Not(nextcube))
            assert(s.check() == unsat)
            core = s.unsat_core()
            core = [str(core[i]) for i in range(0, len(core), 1)] # -> ['p1','p3'], core -> nextcube
        
        # cube_list = []
        # for index, literals in enumerate(tcube_cp.cubeLiterals):
        #     if index in core_list:
        for idx in range(len(tcube_cp.cubeLiterals)):
            var, val = _extract(prev_cube.cubeLiterals[idx])
            data[str(var)] = 0
            if 'p'+str(idx) not in core:
                tcube_cp.cubeLiterals[idx] = True
                data[str(var)] = 1
        # for the time being, completely rely on unsat core reduce



        # tcube_cp.cubeLiterals = cube_list
        #For loop in all previous cube

        # for i in range(len(ground_true.cubeLiterals)):
        #     var, val = _extract(prev_cube.cubeLiterals[i])
        #     data[str(var)] = 0 #TODO: Solve the issue that it contains float or int
        #     assert (type(data[str(var)]) is int)
        #     ground_true.cubeLiterals[i] = Not(ground_true.cubeLiterals[i]) # Flip the variable in f(v1,v2,v3,v4...)
        #     s = Solver()
        #     s.add(ground_true.cube()) #check if the miniterm (state) is sat or not
        #     res = s.check() #check f(v1,v2,v3,v4...) is
        #     #print("The checking result after fliping literal: ",res)
        #     assert (res == sat)
        #     # check the new sat model can transit to the CTI (true means it still can reach CTI)
        #     if str(s.model().eval(nextcube)) == 'True': #TODO: use tenary simulation -> solve the memeory exploration issue
        #         index_to_remove.append(i)
        #         # children = literal.children()
        #         # assert (len(children) == 2)
        #         #
        #         # if str(children[0]) in ['True', 'False']:
        #         #     v = str(children[1])
        #         # elif str(children[1]) in ['True', 'False']:
        #         #     v = str(children[0])
        #         # else:
        #         #     assert (False)
        #         # assert (v[0] in ['i', 'v'])
        #         # if v[0] == 'i':
        #         #     index_to_remove.add(idx)
        #         data[str(var)] = 1
        #         assert (type(data[str(var)]) is int)
        #         # substitute its negative value into nextcube
        #         v, val = _extract(prev_cube.cubeLiterals[i]) #TODO: using unsat core to reduce the literals (as preprocess process), then use ternary simulation
        #         #nextcube = simplify(And(substitute(nextcube, [(v, Not(val))])))
        #         nextcube = simplify(And(substitute(nextcube, [(v, Not(val))]), substitute(nextcube, [(v, val)])))
        #
        #     ground_true.cubeLiterals[i] = prev_cube.cubeLiterals[i]
        #TODO: Compare the ground true and unsat core

        # prev_cube.cubeLiterals = [prev_cube.cubeLiterals[i] for i in range(0, len(prev_cube.cubeLiterals), 1) if i not in index_to_remove]
        # tcube_cp.cubeLiterals = [tcube_cp.cubeLiterals[i] for i in range(0, len(tcube_cp.cubeLiterals), 1) if i not in index_to_remove]
        # return tcube_cp

        if self.smt2_gen_GP==1: self.generaliztion_data_GP.append(data)

        tcube_cp.remove_true()
        #print("After generalization by using unsat core : ",len(tcube_cp.cubeLiterals))
        #print("After generalization by dropping literal one by one : ", len(index_to_remove))
        return tcube_cp

    def solveRelative_RL(self, tcube):
            cubePrime = substitute(substitute(tcube.cube(), self.primeMap),self.inp_map)
            s = Solver()
            s.add(self.frames[tcube.t - 1].cube())
            s.add(self.trans.cube())
            s.add(Not(tcube.cube()))
            s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
            if (s.check() != unsat):  # cube was not blocked, return new tcube containing the model
                model = s.model()
                # c = tCube(tcube.t - 1) #original verison
                # c.addModel(self.lMap, model)  # c = sat_model, original verison
                # return c #original verison
                return tCube(model, self.lMap, tcube.t - 1), None
            else:
                res,h= self.RL(tcube)
                return None, res

    def getBadCube(self):
        print("seek for bad cube...")

        s = Solver() #TODO: the input should also map to input'(prime)
        s.add(substitute(substitute(Not(self.post.cube()), self.primeMap),self.inp_map)) #TODO: Check the correctness here
        s.add(self.frames[-1].cube())
        s.add(self.trans.cube())

        if s.check() == sat: #F[-1] /\ T /\ !P(s') is sat! CTI (cex to induction) found!
            res = tCube(len(self.frames) - 1)
            res.addModel(self.lMap, s.model(), remove_input=False)  # res = sat_model
            print("get bad cube size:", len(res.cubeLiterals), end=' --> ') # Print the result
            # sanity check - why?
            self._debug_c_is_predecessor(res.cube(), self.trans.cube(), True, substitute(substitute(self.post.cube(), self.primeMap),self.inp_map)) #TODO: Here has bug
            new_model = self.generalize_predecessor(res, Not(self.post.cube())) #new_model: predecessor of !P extracted from SAT witness
            print(len(new_model.cubeLiterals)) # Print the result
            self._debug_c_is_predecessor(new_model.cube(), self.trans.cube(), True, substitute(substitute(self.post.cube(), self.primeMap),self.inp_map))
            new_model.remove_input()
            return new_model
        else:
            return None

    def RandL(self, tcube):
        STEPS = self.agent.action_size
        done = False
        M = tcube.M
        orig = np.array([i for i in tcube.M if '\'' not in str(i)])
        cp = np.copy(orig)
        for ti in range(STEPS):
            action = np.random.randint(STEPS) % len(cp)
            cp = np.delete(cp, action);
            cubeprime = substitute(substitute(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.primeMap),self.inp_map)
            s = Solver()
            s.add(Not(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp])))
            s.add(self.R[tcube.t - 1])
            s.add(self.trans.cube())
            s.add(cubeprime)
            start = time.time()
            SAT = s.check();
            interv = time.time() - start
            if SAT != unsat:
                break
            else:
                if (self.isInitial(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.init)):
                    break
                else:
                    orig = np.copy(cp)
        return And(*[self.lMap[str(l)] == M.get_interp(l) for l in orig]), None

    def RL(self, tcube):
        '''
        :param tcube:
        :return: res -> generalized q (q-like) , h -> None
        '''
        STEPS = self.agent.action_size
        # agent.load("./save/cartpole-ddqn.h5")
        done = False
        batch_size = 10
        history_QL = [0]
        state = [-1] * 10
        state = np.reshape(state, [1, self.agent.state_size])

        M = tcube.M
        orig = np.array([i for i in tcube.M if '\'' not in str(i)])
        cp = np.copy(orig)
        for ti in range(STEPS):
            # env.render()
            action = self.agent.act(state) % len(cp) # MLP return back the index of throwing literal
            cp = np.delete(cp, action);
            cubeprime = substitute(substitute(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.primeMap),self.inp_map)
            s = Solver()
            s.add(Not(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp])))
            s.add(self.R[tcube.t - 1])
            s.add(self.trans.cube())
            s.add(cubeprime)
            start = time.time()
            SAT = s.check();
            interv = time.time() - start
            if SAT != unsat:
                reward = -1
                done = True
            else:
                if (self.isInitial(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.init)):
                    reward = 0
                    done = True
                else:
                    reward = max(10 / interv, 1)
                    orig = np.copy(cp)

            next_state = [b for (a, b) in s.statistics()][:-4]
            if (len(next_state) > 10):
                next_state = next_state[0:10]
            else:
                i = len(next_state)
                while (i < 10):
                    next_state = np.append(next_state, -1)
                    i += 1
            # print(next_state)
            history_QL[-1] += reward
            next_state = np.reshape(next_state, [1, self.agent.state_size])
            self.agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                history_QL.append(0)
                self.agent.update_target_model()
                break
            if len(self.agent.memory) > batch_size:
                self.agent.replay(batch_size)
        # And(*[self.lMap[str(l)] == M.get_interp(l) for l in orig])-> generlization (when unsat core not exists)
        tmp_cube = And(*[self.lMap[str(l)] == M.get_interp(l) for l in orig])
        return And(*[self.lMap[str(l)] == M.get_interp(l) for l in orig]), history_QL

    def _debug_trace(self, trace: PriorityQueue):
        prev_fidx = 0
        self.bmc.setup()
        while not trace.empty():
            idx, cube = trace.get()
            assert (idx == prev_fidx+1)
            self.bmc.unroll()
            self.bmc.add(cube.cube())
            reachable = self.bmc.check()
            if reachable:
                print (f'F {prev_fidx} ---> {idx}')
            else:
                print(f'F {prev_fidx} -/-> {idx}')
                assert(False)
            prev_fidx += 1
        self.bmc.unroll()
        self.bmc.add(Not(self.post.cube()))
        assert(self.bmc.check() == sat)


    def _sanity_check_inv(self, inv):
        pass

    def _sanity_check_frame(self):
        for idx in range(0,len(self.frames)-1):
            # check Fi => Fi+1
            # Fi/\T => Fi+1
            Fi = self.frames[idx].cube()
            Fiadd1 = self.frames[idx+1].cube()
            s1 = Solver()
            s1.add(Fi)
            s1.add(Not(Fiadd1))
            assert( s1.check() == unsat)
            s2 = Solver()
            s2.add(Fi)
            s2.add(self.trans.cube())
            s2.add(substitute(substitute(Not(Fiadd1), self.primeMap),self.inp_map))
            assert( s2.check() == unsat)




if __name__ == '__main__':
    pass
