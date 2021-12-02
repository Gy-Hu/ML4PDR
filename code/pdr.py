from z3 import *
import time
import sys
import numpy as np
import copy
from line_profiler import LineProfiler
from functools import wraps

# 查询接口中每行代码执行的时间
def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        func_return = f(*args, **kwargs)
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        lp.print_stats()
        return func_return
    return decorator

#TODO: Using Z3 to check the 3 properties of init, trans, safe, inductive invariant

# conjunction of literals.
class tCube:
    # make a tcube object assosciated with frame t.
    def __init__(self, t=0):
        self.t = t
        self.cubeLiterals = list()

    def clone(self):
        ret = tCube(self.t)
        ret.cubeLiterals = self.cubeLiterals.copy()
        return ret

#TODO: Using multiple timer to caculate which part of the code has the most time consumption
    # 解析 sat 求解出的 model, 并将其加入到当前 tCube 中
    def addModel(self, lMap, model, remove_input = True):
        no_primes = [l for l in model if str(l)[-1] != '\'']
        if remove_input:
            no_input = [l for l in no_primes if str(l)[0] != 'i']
        else:
            no_input = no_primes
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

    # 删除第 i 个元素，并返回 tCube
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
    #
    # def cube(self):
    #     return And(*self.cubeLiterals)

    def __repr__(self):
        return str(self.t) + ": " + str(sorted(self.cubeLiterals, key=str))


# class tClause:
#     def __init__(self, t=0):
#         self.t = t
#         self.clauseLiterals = []
#
#     def defFromNotCube(self, c: tCube):
#         for cube in c.cubeLiterals:
#             self.clauseLiterals.append(Not(cube))
#
#     def clause(self):
#         return simplify(Or(self.clauseLiterals))
#
#     def add(self, m):
#         self.clauseLiterals.append(m)
#
#     def delete(self, i: int):
#         res = tClause(self.t)
#         for it in range(len(self.clauseLiterals)):
#             if it == i:
#                 continue
#             res.add(self.clauseLiterals[it])
#         return res
#
#     def __repr__(self):
#         return str(self.t) + ": " + str(sorted(self.clauseLiterals, key=str))


class PDR:
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
        self.literals = literals
        self.items = self.primary_inputs + self.literals
        self.lMap = {str(l): l for l in self.items}
        self.post = post
        self.frames = list()
        self.primeMap = [(literals[i], primes[i]) for i in range(len(literals))]
        self.pv2next = pv2next
        # print("init:")
        # print(self.init.cube())
        # print("trans...")
        # print(self.trans.cube())
        # print("post:")
        # print(self.post.cube())

    def run(self,agent):
        self.agent = agent
        self.frames = list()
        self.frames.append(self.init)

        while True:
            c = self.getBadCube()
            if c is not None:
                # print("get bad cube!")
                trace = self.recBlockCube(c) #TODO: 找出spec3-and-env这个case为什么没有recBlock
                if trace is not None:
                    print("Found trace ending in bad state:")
                    for f in trace:
                        print(f)
                    return False
                print("recBlockCube Ok! F:")
                # for i in self.frames:
                #     print(i)
            else:
                inv = self.checkForInduction(self.frames)
                if inv != None:
                    print("Found inductive invariant:", simplify(inv.cube()))
                    return True
                print("Did not find invariant, adding frame " + str(len(self.frames)) + "...")

                # self.frames.append(tCube(len(self.frames))) #TODO: Optimize the way of adding frames
                # self.frames[-1].add(True)

                # tran = copy.deepcopy(self.trans)
                # tran.t = len(self.frames)
                # self.frames.append(tran)
                #print("print the frame")

                print("Adding frame " + str(len(self.frames)) + "...")
                self.frames.append(tCube(len(self.frames))) #TODO: Append P, and get bad cube change to F[-1] /\ T /\ !P' (also can do generalization), check it is sat or not
                # [init, P]
                # init /\ bad   ?sat
                # init /\T /\ bad'  ?sat

                print("Now print out the size of frames")
                for index in range(len(self.frames)):
                    print("F", index , 'size:' , len(self.frames[index].cubeLiterals))

                #TODO: Try new way to pushing lemma (like picking >=2 clause at once to add in new frame)
                fi = self.frames[-2]
                for c in fi.cubeLiterals: # Pushing lemma = propagate clause
                    s = Solver()
                    s.add(fi.cube())
                    s.add(self.trans.cube())
                    s.add(Not(substitute(c, self.primeMap)))  # F[i] and T and Not(c)'
                    if s.check() == unsat:
                        self.frames[-1].add(c)

                # for index, fi in enumerate(self.frames): #TODO: Solve the issue that here is quite slow!!
                #     if index == len(self.frames) - 1:
                #         break
                #     print("F", index , 'size:' , len(fi.cubeLiterals))
                #     for c in fi.cubeLiterals: # Pushing lemma = propagate clause
                #         s = Solver()
                #         s.add(fi.cube())#TODO: Record which literal has been pushed, delete duplicated literals
                #         s.add(self.trans.cube())
                #         s.add(Not(substitute(c, self.primeMap)))  # F[i] and T and Not(c)'
                #         if s.check() == unsat:
                #             self.frames[index + 1].add(c) #TODO: Modify here to append safety property

                #     if self.checkForInduction(fi):
                #         print("Fond inductive invariant:\n" + str(fi.cube()))
                #         #print("Fond inductive invariant:\n")
                #         return True
                #print("New Frame " + str(len(self.frames) - 1) + ": ")
                #print(self.frames[-1].cube())

    # def checkForInduction(self, frame):
    #     print("check for Induction now...")
    #     s = Solver()
    #     s.add(self.trans.cube())
    #     s.add(frame.cube())
    #     s.add(Not(substitute(frame.cube(), self.primeMap)))  # T and F[i] and Not(F[i])'
    #     if s.check() == unsat:
    #         return True
    #     return False

    def checkForInduction(self, frame):
        #print("check for Induction now...")
        for frame in self.frames:
            s = Solver()
            s.add(self.trans.cube())
            s.add(frame.cube())
            s.add(Not(substitute(frame.cube(), self.primeMap)))  # T and F[i] and Not(F[i])'
            if s.check() == unsat:
                return frame
        return None
            #     return True
            # return False

    #TODO: 解决这边特殊case遇到safe判断成unsafe的问题
    def recBlockCube(self, s0: tCube):
        print("recBlockCube now...") #TODO: Maybe there's a bug here when rec to F[0]? Find it! （考虑MIC的极端情况，或者generalized predecessor的极端情况）
        Q = [s0]
        while len(Q) > 0:
            s = Q[-1]
            if s.t == 0: #TODO: enhance counter-example finding here (read ic3, pdr & friend)
                return Q
            z = self.solveRelative(s)
            if z is None:
                Q.pop()
                s = self.MIC(s)
                for i in range(1, s.t + 1):
                    self.frames[i].add(Not(s.cube())) #TODO: Try RL here
            else: #SAT condition
                Q.append(z) #TODO: Add ternary simulation to generalize predecessor (Gneralize CTI)
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

    def MIC(self, q: tCube): #TODO: Check the algorithm is correct or not
        for i in range(len(q.cubeLiterals)):
            q1 = q.delete(i)
            if self.down(q1):
                q = q1
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

    def down(self, q: tCube):
        while True:
            s = Solver()
            s.push()
            #s.add(And(self.frames[0].cube(), Not(q.cube())))
            s.add(And(self.frames[0].cube(), (q.cube())))
            #if unsat == s.check():
            if sat == s.check():
                return False
            s.pop()
            s.push()
            s.add(And(self.frames[q.t].cube(), Not(q.cube()), self.trans.cube(),
                      substitute(q.cube(), self.primeMap)))  # Fi and not(q) and T and q'
            if unsat == s.check():
                return True
            # m = s.model()
            # q.addModel(self.lMap, m)
            # s.pop()
            return False

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

    # tcube is bad state
    def solveRelative(self, tcube):
        cubePrime = substitute(tcube.cube(), self.primeMap)
        s = Solver()
        s.add(Not(tcube.cube()))
        s.add(self.frames[tcube.t - 1].cube())
        s.add(self.trans.cube())
        s.add(cubePrime)  # F[i - 1] and T and Not(badCube) and badCube'
        if s.check() == sat:
            model = s.model()
            c = tCube(tcube.t - 1)
            c.addModel(self.lMap, model, remove_input=False)  # c = sat_model
            #return c
            print("original cube length: ", len(c.cubeLiterals))
            generalized_p = self.generalize_predecessor(c, tcube)
            print("generalized cube length: ", len(generalized_p.cubeLiterals))
            # remove input
            generalized_p.remove_input()
            return generalized_p #TODO: Using z3 eval() to conduct tenary simulation
        return None
#TODO: Get bad cude should generalize as well!
    def generalize_predecessor(self, prev_cube:tCube, next_cube):
        #check = tcube.cube()
        tcube_cp = prev_cube.clone() #TODO: Solve the z3 exception wranning
        #print("Begin to generalize predessor")
        nextcube = substitute(substitute(next_cube.cube(), self.primeMap), list(self.pv2next.items()))

        index_to_remove = []

        s = Solver()
        s.add(prev_cube.cube())
        s.check()
        assert(str(s.model().eval(nextcube)) == 'True')

        for i in range(len(tcube_cp.cubeLiterals)):
            #print("Now begin to check the No.",i," of cex")
            tcube_cp.cubeLiterals[i] = Not(tcube_cp.cubeLiterals[i])
            s = Solver()
            s.add(tcube_cp.cube())
            res = s.check()
            assert (res == sat)
            if str(s.model().eval(nextcube)) == 'True':
                index_to_remove.append(i)
            tcube_cp.cubeLiterals[i] = prev_cube.cubeLiterals[i]

            # cubePrime = substitute(tcube_cp.cube(), self.primeMap)
            # s = Solver()
            # s.add(Not(tcube_cp.cube()))
            # s.add(self.frames[tcube_cp.t - 1].cube())
            # s.add(self.trans.cube())
            # s.add(cubePrime)
            # if s.check() == sat:
            #     #print("found way to generalize the predessor!")
            #     #print(check)
            #     index_to_remove.append(i)
            #     # print("Before:",tcube.cubeLiterals)
            #     # tcube.cubeLiterals.pop(i)
            #     # print("After:",tcube.cubeLiterals)

        prev_cube.cubeLiterals = [prev_cube.cubeLiterals[i] for i in range(0, len(prev_cube.cubeLiterals), 1) if i not in index_to_remove]
        return prev_cube

    def solveRelative_RL(self, tcube):
            cubePrime = substitute(tcube.cube(), self.primeMap)
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
        model = And(Not(self.post.cube()), self.frames[-1].cube())  # F[k] and Not(p)
        s = Solver()
        s.add(model)
        if s.check() == sat:
            res = tCube(len(self.frames) - 1)
            res.addModel(self.lMap, s.model())  # res = sat_model
            print("get bad cube:")
            #print(res.cube()) # Print the result
            return res
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
            cubeprime = substitute(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.primeMap)
            s = Solver()
            s.add(Not(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp])))
            s.add(self.R[tcube.t - 1])
            s.add(self.trans)
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
            cubeprime = substitute(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp]), self.primeMap)
            s = Solver()
            s.add(Not(And(*[self.lMap[str(l)] == M.get_interp(l) for l in cp])))
            s.add(self.R[tcube.t - 1])
            s.add(self.trans)
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


if __name__ == '__main__':
    pass
