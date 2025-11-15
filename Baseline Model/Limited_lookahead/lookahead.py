import random
from sklearn.naive_bayes import MultinomialNB
import time
import warnings

warnings.filterwarnings('ignore')
import copy
import os

from option_sampler import OptionSampler
from ao_tree import aoTree

class BeliefState:
    __slots__ = ['t', 'history', 'N']

    def __init__(self, t):
        self.t = t
        self.history = None
        self.N = 0

def sampleBeliefObservation(option, clfArr, b, b_next):
    b_next.history = copy.deepcopy(b.history)
    b_next.history[option].pop(0)

    # obs = int(clfArr[option].predict([b.history[option]])[0])
    # b_next.history[option].append(obs) # obs = 0:'-' / 1:'+'
    # return obs

    pr_ML_PosiObs = clfArr[option].predict_proba([b.history[option]])[0]
    if clfArr[option].classes_[0]:  # classes_: [0 1] or [0] or [1]
        pr_ML_PosiObs = pr_ML_PosiObs[0]
    else:
        pr_ML_PosiObs = 1 - pr_ML_PosiObs[0]
    # pr_ML_PosiObs = np.isnan(pr_ML_PosiObs) and 1 or pr_ML_PosiObs
    rand = random.random()
    if pr_ML_PosiObs <= rand:
        b_next.history[option].append(0)  # represent '-' observation
        return 0
    b_next.history[option].append(1)  # represent '+' observation
    return 1

def sampleBeliefState(b, os_, aot, clfArr, t, look_step = 3):
    # option = random.randint(0, aot.numOptions-1) # or comment os

    option = os_.sample()
    b_next = BeliefState(t)

    aot_temp1 = copy.deepcopy(aot)
    aot_temp2 = copy.deepcopy(aot)

    for i in range(look_step):
        obs = sampleBeliefObservation(option, clfArr, b, b_next)
        # aot.update()调用后会更新树，固定深度的前瞻算法通过模拟，选择根节点值最大的
        aot_temp1.update(option, obs)
        if aot_temp2.root.prob < aot_temp1.root.prob:
            aot_temp2 = copy.deepcopy(aot_temp1)

    aot = copy.deepcopy(aot_temp2)
    os_.next(option, obs)

    # print('option=',option,'obs=',obs)
    # print('cost=',aot.root.cost,'prob',aot.root.prob)
    return (b_next, aot)

def calculateV(b, aot, t, VArr, NArr):
    NArr[t] += 1
    VArr[t] = (VArr[t] * (NArr[t] - 1) + aot.root.prob) / NArr[t]
    # VArr[t] = (VArr[t] * (NArr[t] - 1) + \
    # 	aot.root.cost) / NArr[t]
    return (VArr, NArr)

def sampleTrajectory(b, os_, aot, clfArr, t, hor, VArr, NArr):
    b.N += 1
    VArr, NArr = calculateV(b, aot, t, VArr, NArr)
    if t < hor:
        t += 1
        b_next, aot = sampleBeliefState(b, os_, aot, clfArr, t)
        VArr, NArr = sampleTrajectory(b_next, os_, aot, clfArr, t, hor, VArr, NArr)
    return (VArr, NArr)

def MC_method(b0, os_, aot, clfArr, hor, itr, VArr, NArr):
    while itr:
        # print("Iteration =",itr)
        os_.reset()
        aot.reset()
        VArr, NArr = sampleTrajectory(b0, os_, aot, clfArr, 0, hor, VArr, NArr)
        # print("VArr =",VArr)
        # print("NArr =",NArr)
        itr -= 1
    return (VArr, NArr)

def integrateHis(s, dim):
    source = [int(i) for i in s.split(',')]
    if len(source) < dim:
        dim = len(source)

    # init b0.history[option]
    history = source[:dim - 1]

    # sampleData
    X = []
    Y = []
    sampleSize = dim * dim
    for i in range(sampleSize):
        X.append(random.sample(source, dim - 1))
        Y.append(random.sample(source, 1)[0])

    # generateClf
    clf = MultinomialNB()
    clf.fit(X, Y)
    return (history, clf)


def write_to_file(file_path, content):
    # 确保目录存在
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 以追加模式打开文件，如果文件不存在则创建文件
    with open(file_path, 'a') as file:
        file.write(content + '\n')  # 添加内容并换行


if __name__ == '__main__':
    hor = 80
    itr = 200

    VArr = [0 for i in range(hor + 1)]  # VArr[t]
    NArr = [0 for i in range(hor + 1)]  # NArr[t]
    dftDim = 18

    fileHeader = '../Synthetic_data_mc_60/'

    # and-or Tree结构
    exps_N_Filename = fileHeader + 'aoTree3.txt'
    historyFilename = fileHeader + 'sys_data_history2.txt'

    exps_N_File = open(exps_N_Filename)
    tmpArr = []
    for line in exps_N_File:
        tmpArr.append(line)
    exps = tmpArr[0]
    numOptions = int(tmpArr[1])
    optionOrder = list(map(int, tmpArr[2][:-1].split(',')))
    exps_N_File.close()

    b0 = BeliefState(0)

    tmpArr.clear()
    historyFile = open(historyFilename)
    for line in historyFile:
        if '-1\n' == line:
            continue
        tmpArr.append(integrateHis(line, dftDim))
    historyFile.close()
    b0.history = [None] * numOptions
    clfArr = [None] * numOptions
    for option in optionOrder:
        b0.history[option] = tmpArr[option][0]
        clfArr[option] = tmpArr[option][1]

    os_ = OptionSampler(exps, optionOrder)
    aot = aoTree(exps, numOptions)
    # aot.reset()
    # print('bb cost=',aot.root.cost,'prob',aot.root.prob)

    runtime0 = time.time()
    VArr, NArr = MC_method(b0, os_, aot, clfArr, hor, itr, VArr, NArr)
    print("Program execution time:", time.time() - runtime0)
    print('VArr =', VArr)
    print("VArr len:", len(VArr))

    # print('NArr =', NArr)

    # min_v = min(VArr)
    # max_v = max(VArr)
    # VArr = list(map(lambda x: (x-min_v)/(max_v-min_v), VArr))
    # print('\nnorm_VArr=\n', VArr)

    # print('\nos.count=', os.count, 'ave(os.NOptionsArr)=', sum(os.NOptionsArr)/len(os.NOptionsArr))

    # # == [0] ==
    # C = 1e-3
    # acc_C = C * hor # acc: accumulated
    # stop = hor
    # VArr[hor] -= acc_C
    # EV0 = VArr[hor]
    # for t in range(hor-1, 0, -1):
    # 	acc_C = round(acc_C - C, 3) # 3 because 1e-3
    # 	VArr[t] -= acc_C
    # 	if VArr[t] >= EV0:
    # 		EV0 = VArr[t]
    # 		stop = t
    # print('EV(0) =', EV0)
    # print('stop =', stop)

    # == [1] ==
    C_Arr = [1e-2, 0.7e-2, 0.4e-2, \
             1e-3, 0.7e-3, 0.4e-3, \
             1e-4, 0.7e-4, 0.4e-4]
    V_negC_Arr = []
    # EV0的个数和C_Arr的个数有关，计划处理多少个工人成本，EV0就有多少个
    EV0_Arr = []
    stop_Arr = []
    # 简单理解，用VArr最后一个元素 减去 当前循环到的工人成本*hor，得到处理后的expected value
    for C in C_Arr:
        tmpVArr = VArr.copy()
        # 一次的成本为C，总共hor次
        acc_C = C * hor  # acc: accumulated
        stop = hor
        tmpVArr[hor] -= acc_C
        EV0 = tmpVArr[hor]
        for t in range(hor - 1, 0, -1):
            acc_C = round(acc_C - C, 5)  # 3 because 1e-5
            tmpVArr[t] -= acc_C
            if tmpVArr[t] >= EV0:
                EV0 = tmpVArr[t]
                stop = t
        V_negC_Arr.append(tmpVArr)
        EV0_Arr.append(EV0)
        # 不同工人成本下的停止时间
        stop_Arr.append(stop)
    # print('\nV_negC_Arr =')
    # for i,C in enumerate(C_Arr):
    # 	print('round =', i, 'C =', C, 'V_negC =\n', V_negC_Arr[i], '\n')
    print('EV0_Arr =', EV0_Arr)
    print('stop_Arr =', stop_Arr)

# prob = VArr[stop_Arr[3]]
# print('prob of stop_Arr[\'1e-3\']', prob)
# if '1' == exps_N_Filename[1]:
# 	print( 'case 1:', prob * 4 + (1 - prob) * 5 )
# else:
# 	print( 'case 2 or 3:', prob * 4 + (1 - prob) * 12 )