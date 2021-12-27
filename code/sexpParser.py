from pyparsing import *
from base64 import b64decode
import pprint
import networkx as nx
import matplotlib.pyplot as plt
#from past.builtins import basestring

def build(g, X):
    if isinstance(X, list):
        parent = X[0]
        g.add_node(parent)
        for branch in X[1:]:
            child = build(g, branch)
            g.add_edge(parent, child)

        return parent

    if isinstance(X, str):
        g.add_node(X)
        return X

def verifyLen(s,l,t):
    t = t[0]
    if t.len is not None:
        t1len = len(t[1])
        if t1len != t.len:
            raise ParseFatalException(s,l,\
                    "invalid data of length %d, expected %s" % (t1len, t.len))
    return t[1]

# define punctuation literals
LPAR, RPAR, LBRK, RBRK, LBRC, RBRC, VBAR = map(Suppress, "()[]{}|")

decimal = Regex(r'0|[1-9]\d*').setParseAction(lambda t: int(t[0]))
hexadecimal = ("#" + OneOrMore(Word(hexnums)) + "#")\
                .setParseAction(lambda t: int("".join(t[1:-1]),16))
bytes = Word(printables)
raw = Group(decimal("len") + Suppress(":") + bytes).setParseAction(verifyLen)
token = Word(alphanums + "-./_:*+=")
base64_ = Group(Optional(decimal|hexadecimal,default=None)("len") + VBAR
    + OneOrMore(Word( alphanums +"+/=" )).setParseAction(lambda t: b64decode("".join(t)))
    + VBAR).setParseAction(verifyLen)

qString = Group(Optional(decimal,default=None)("len") +
                        dblQuotedString.setParseAction(removeQuotes)).setParseAction(verifyLen)
simpleString = base64_ | raw | decimal | token | hexadecimal | qString

# extended definitions
decimal = Regex(r'-?0|[1-9]\d*').setParseAction(lambda t: int(t[0]))
real = Regex(r"[+-]?\d+\.\d*([eE][+-]?\d+)?").setParseAction(lambda tokens: float(tokens[0]))
token = Word(alphanums + "-./_:*+=!<>")

simpleString = real | base64_ | raw | decimal | token | hexadecimal | qString

display = LBRK + simpleString + RBRK
string_ = Optional(display) + simpleString

sexp = Forward()
sexpList = Group(LPAR + ZeroOrMore(sexp) + RPAR)
sexp << ( string_ | sexpList )

######### Test data ###########
test00 = """
    (let ((a!1 (and (not (not v14)) (not (and v30 (not v28))) i10))
      (a!2 (and v22 (not (and (not v18) (not v16))) i4))
      (a!3 (and v18 (not (and (not v14) (not v12))) i2))
      (a!4 (and v26 (not (and (not v22) (not v20))) i6))
      (a!5 (and (not v30) (not (and (not v26) (not v24))) i8)))
    (let ((a!6 (and (not (and (not a!1)
                              (not a!2)
                              (not a!3)
                              (not a!4)
                              (not a!5)
                              (not i10_prime)
                              (not i8_prime)
                              (not i6_prime)
                              (not i4_prime)))
                    (not a!3)))
          (a!7 (and (not (and (not a!1)
                              (not a!2)
                              (not a!3)
                              (not a!4)
                              (not a!5)
                              (not i10_prime)
                              (not i8_prime)
                              (not i6_prime)))
                    (not a!2)))
          (a!8 (and (not (and (not a!1)
                              (not a!2)
                              (not a!3)
                              (not a!4)
                              (not a!5)
                              (not i10_prime)
                              (not i8_prime)))
                    (not a!4)))
          (a!9 (and (not (and (not a!1)
                              (not a!2)
                              (not a!3)
                              (not a!4)
                              (not a!5)
                              (not i10_prime)))
                    (not a!5)))
          (a!10 (and (not a!1) (not (and (not a!2) (not a!3) (not a!4) (not a!5))))))
    (let ((a!11 (and (not (and (not a!6) i2_prime (not a!7) i4_prime))
                     (not (and (not a!6) i2_prime (not a!8) i6_prime))
                     (not (and (not a!6) i2_prime (not a!9) i8_prime))
                     (not (and (not a!6) i2_prime (not a!10) i10_prime))
                     (not (and (not a!7) i4_prime (not a!8) i6_prime))
                     (not (and (not a!7) i4_prime (not a!9) i8_prime))
                     (not (and (not a!7) i4_prime (not a!10) i10_prime))
                     (not (and (not a!8) i6_prime (not a!9) i8_prime))
                     (not (and (not a!8) i6_prime (not a!10) i10_prime))
                     (not (and (not a!10) i10_prime (not a!9) i8_prime)))))
    (not a!11))))
    """

if __name__ == '__main__':
  # Run tests
    sexpr = sexp.parseString(test00, parseAll=True)
    pprint.pprint(sexpr.asList())
    #-- Get the parsing results as a list of component lists.
    nested = sexpr.asList()
    print("nested is ", nested)