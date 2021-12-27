import z3


OP_VAR=0
OP_NOT=1
OP_AND=2
OP_CONST=3

_TRUE=(0,1)
_FALSE=(1,0)
_X = (1,1)
_BOT = (0,0)

def encode(v):
    if str(v) == 'True':
        return _TRUE
    if str(v) == 'False':
        return _FALSE
    assert False

class AIGBuffer(object):
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
        self.expr_to_item = dict()   # expr -> item_no
        self.item_to_expr = dict()   # item_no -> expr
        self.item_use_list = dict()  # item_no -> [uses]
        self.vname_to_vid = dict()  # str -> nid
        self.items = []
        # items: a list of the followings
        #  (AND, [item_number])
        #  (NOT, [item_number])
    def clone(self):
        ret = AIGBuffer()
        ret.expr_to_item = self.expr_to_item.copy()
        ret.item_to_expr = self.item_to_expr.copy()
        ret.item_use_list = self.item_use_list.copy()
        ret.vname_to_vid = self.vname_to_vid.copy()
        ret.items = self.items.copy()
        return ret

    def _new_item_number(self):
        l = len(self.items)
        self.items.append(tuple())
        return l

    def register_expr(self, expr):
        if expr in self.expr_to_item:
            return self.expr_to_item[expr]  # remember to incr reference in the item
        item_no = self._new_item_number()
        self.expr_to_item[expr] = item_no
        self.item_to_expr[item_no] = expr
        op = expr.decl().kind()
        children = expr.children()
        if len(children) == 0:
            # it is a variable
            if str(expr) in ['True', 'False']:
                self.items[item_no] = (OP_CONST, [str(expr)])
            else:
                self.items[item_no] = (OP_VAR, [str(expr)])
                self.vname_to_vid[str(expr)] = item_no
        elif op == z3.Z3_OP_NOT:
            assert len(children) == 1
            child_item_num = self.register_expr(children[0])
            self.item_use_list[child_item_num] = self.item_use_list.get(child_item_num, []) + [item_no]
            self.items[item_no] = (OP_NOT, [child_item_num])
        elif op == z3.Z3_OP_AND:
            assert len(children) >= 1
            children_item_no = []
            for c in children:
                child_item_num = self.register_expr(c)
                children_item_no.append(child_item_num)
                self.item_use_list[child_item_num] = self.item_use_list.get(child_item_num, []) + [item_no]
            self.items[item_no] = (OP_AND, children_item_no)
        else:
            assert False
        return item_no

    # (a[0], a[1])
    # True (0,1)
    # False (1,0)
    # X (1,1)
    # bot (0,0)
    @staticmethod
    def _NOT(a):
        return (a[1],a[0])
    @staticmethod
    def _AND(a,b):
        return ( a[0] | b[0] , a[1] & b[1])
    @staticmethod
    def _OR(a,b):
        return (a[0] & b[0] , a[1] | b[1])
    @staticmethod
    def _NOT_bot(a):
        return a[0] or a[1]
    @staticmethod
    def _NOT_X(a):
        return not (a[0] and a[1])
    @staticmethod
    def interpret(a):
        if a == _TRUE:
            return "1"
        if a == _FALSE:
            return "0"
        if a == _X:
            return "X"
        if a == _BOT:
            return "_"
        assert False

    def set_initial_var_assignment(self, model):
        v_assignments = {}
        for v in model:
            val = model[v]
            v_assignments[str(v)] = (_TRUE if str(val) == 'True' else _FALSE)

        self.item_assignments = [None]*len(self.items)
        idstack = list(range(len(self.items)))
        while len(idstack) != 0:
            nid = idstack[-1]
            if self.item_assignments[nid] is not None:
                del idstack[-1]
                continue
            op, children = self.items[nid]
            if op == OP_CONST:
                self.item_assignments[nid] = encode(children[0])
                del idstack[-1]
                continue
            elif op == OP_VAR:
                vname = children[0]
                if vname in v_assignments:
                    self.item_assignments[nid] = v_assignments[vname]
                else:
                    self.item_assignments[nid] = _X
                del idstack[-1]
                continue
            # else op == OP_AND or OP_NOT
            can_eval_parent = True
            for c in children:
                if self.item_assignments[c] is None:
                    idstack.append(c)
                    can_eval_parent = False
                    break
            if not can_eval_parent:
                continue
            self.item_assignments[nid] = self._compute(nid)
            del idstack[-1]


    def _compute(self, nid):
        op, children = self.items[nid]
        if op == OP_CONST:
            return encode(children[0])
        elif op == OP_VAR:
            return self.item_assignments[nid]
        elif op == OP_NOT:
            cnid = children[0]
            # assert cnid > nid # may not hold
            return self._NOT(self.item_assignments[cnid])
        elif op == OP_AND:
            cvals = [self.item_assignments[cnid] for cnid in children]
            v = cvals[0]
            for idx in range(1, len(cvals)):
                v = self._AND(v, cvals[idx])
            return v

    def set_Li(self, var, value):
        vid = self.vname_to_vid[str(var)]
        assert isinstance(value, tuple)
        assert value == _TRUE or value == _FALSE or value == _X
        self.item_assignments[vid] = value

        Q = [vid]
        while len(Q) != 0:
            nv = Q[0]
            del Q[0]
            if nv not in self.item_use_list:
                continue
            to_eval = self.item_use_list[nv]
            for n in to_eval:
                old_value = self.item_assignments[n]
                res = self._compute(n)
                self.item_assignments[n] = res
                if res != old_value:
                    Q.append(n)

    def get_val(self, expr):
        return self.item_assignments[self.expr_to_item[expr]]


def test0():
    assert ( AIGBuffer._AND(_TRUE, _TRUE) == _TRUE )
    assert ( AIGBuffer._AND(_TRUE, _FALSE) == _FALSE )
    assert ( AIGBuffer._AND(_TRUE, _X) == _X )
    assert ( AIGBuffer._AND(_FALSE, _TRUE) == _FALSE )
    assert ( AIGBuffer._AND(_FALSE, _FALSE) == _FALSE )
    assert ( AIGBuffer._AND(_FALSE, _X) == _FALSE )

    assert ( AIGBuffer._OR(_TRUE, _TRUE) == _TRUE )
    assert ( AIGBuffer._OR(_TRUE, _FALSE) == _TRUE )
    assert ( AIGBuffer._OR(_TRUE, _X) == _TRUE )
    assert ( AIGBuffer._OR(_FALSE, _TRUE) == _TRUE )
    assert ( AIGBuffer._OR(_FALSE, _FALSE) == _FALSE )
    assert ( AIGBuffer._OR(_FALSE, _X) == _X )

    assert ( AIGBuffer._NOT(_FALSE) == _TRUE )
    assert ( AIGBuffer._NOT(_TRUE) == _FALSE )
    assert ( AIGBuffer._NOT(_X) == _X )


def test():
    a = z3.Bool('a')
    b = z3.Bool('b')
    expr = z3.Not(z3.And(z3.Not(a), z3.Not(b), z3.And(True)))
    print (expr)
    slv = z3.Solver()
    slv.add(expr)
    assert slv.check() == z3.sat
    m = slv.model()
    print (m)
    aigbuf = AIGBuffer()        # new object
    aigbuf.register_expr(expr)  # register the next cube
    aigbuf.set_initial_var_assignment(m)  # set initial model
    assert aigbuf.get_val(expr) == _TRUE
    aigbuf.set_Li( a, _X )
    print ( aigbuf.interpret( aigbuf.get_val(expr) ) )
    aigbuf.set_Li( b, _X )
    print ( aigbuf.interpret(aigbuf.get_val(expr) ) )
    aigbuf.set_Li( b, _TRUE )
    print ( aigbuf.interpret(aigbuf.get_val(expr) ) )
    aigbuf.set_Li( b, _FALSE )
    print ( aigbuf.interpret(aigbuf.get_val(expr) ) )
    aigbuf.set_Li( a, _TRUE )
    print ( aigbuf.interpret(aigbuf.get_val(expr) ) )
    aigbuf.set_Li( b, _X )
    print (aigbuf.interpret( aigbuf.get_val(expr) ) )
    aigbuf.set_Li( a, _FALSE )
    aigbuf.set_Li( b, _FALSE )
    print (aigbuf.interpret( aigbuf.get_val(expr) ) )

if __name__ == '__main__':
    test0()
    test()


        
        
        

