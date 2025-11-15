
class Node:
    __slots__ = ['C', 'Vcoll', 'Vnotcoll', 'N']
    # C 表示采取行动C的预期直接成本
    # Vcoll 表示采取行动C的预期价值
    # Vnotcoll表示采取行动 not C的预期价值
    # N表示到达当前状态的次数

    def __init__(self, ):
        self.Vnotcoll = 0
        self.C = 0
        self.Vcoll = 0
        self.N = 0


def sampleTrueState(bt):
    return 1

def sampleNextBeliefState(bt):
    return 1

def isTerminal(bt,l):
    return 0

def sampleExecutionPath(bt,l):
    if not isTerminal(bt,l):
        bt_new = sampleNextBeliefState(bt)
        s = sampleExecutionPath(bt_new,l)
    else:
        s = sampleTrueState(bt)
    evaluate(bt,s,l)
    return s

def calculateVOI(b0,l,iter):
    for i in range(iter):
        sampleExecutionPath(b0,l)
    voi = b0.Vcoll-b0.Vnotcoll
    return voi

def R_s_d(s,d_):
    return 1

def R_s_c(s,c):
    return 1

def d_star(bt):
    return 1

def evaluate(bt,s,l):
    bt.Nnotcoll = bt.Nnotcoll+1
    bt.Vnotcoll = (bt.Vnotcoll*(bt.Nnotcoll - 1) + R_s_d(s,d_star(bt)))/bt.Nnotcoll
    if not isTerminal(bt,l):
        bt.Ncoll = 00000000000000000000
        # 可以不考虑成本，就算是忽略成本了
        bt.C = 0
        bt.Vcoll = 00000000

    if bt.Vnotcoll >= bt.Vcoll or bt.Ncoll == 0:
        bt.V,bt.N = bt.Vnotcoll,bt.Nnotcoll
    else:
        bt.V,bt.N = bt.Vcoll,bt.Ncoll
