import random
from sklearn.naive_bayes import MultinomialNB
import time
import warnings

warnings.filterwarnings('ignore')
import copy
import os
import itertools

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
    # print("b.history =",b.history,"b_next.history =",b_next.history)

    # print("b_next.history[option] =",b_next.history[option])
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
    # option = os_.sample()

    b_next = BeliefState(t)
    # print("b_next =",b_next)

    option_unuse = list(set(optionOrder_global) - set(option_use))
    option_use_temp = copy.deepcopy(option_use)
    option_global_temp = copy.deepcopy(optionOrder_global)

    if look_step > len(option_unuse):
        possible_option_list = option_unuse
    else:
        possible_option_list = list(itertools.combinations(option_unuse, look_step))
        for j in range(len(possible_option_list)):
            possible_option_list[j] = list(possible_option_list[j])

    aot_temp2 = copy.deepcopy(aot)
    b_next_temp1 = copy.deepcopy(b_next)
    b_next_temp2 = copy.deepcopy(b_next)
    # print("b_next_temp =",b_next_temp)
    obs_backup = 0

    # print("len(possible_option_list) =",len(possible_option_list))
    for l in range(len(possible_option_list)):
        # print("l =",l)
        # print(possible_option_list[l])
        aot_temp1 = copy.deepcopy(aot)
        obs_temp = 0
        for r in range(len(possible_option_list[l])):
            # print("possible_option_list[l][r] =",possible_option_list[l][r])
            obs_temp = sampleBeliefObservation(possible_option_list[l][r], clfArr, b, b_next_temp1)
            # 参数为option、obs，option原来是采样来的，现在使用固定的
            aot_temp1.update(possible_option_list[l][r],obs_temp)

        if aot_temp2.root.prob < aot_temp1.root.prob:
            aot_temp2 = copy.deepcopy(aot_temp1)
            obs_backup = copy.deepcopy(obs_temp)
            b_next_temp2 = copy.deepcopy(b_next_temp1)

        b_next_temp1 = copy.deepcopy(b_next)

    # aot_temp2中记录的是最大值对应的树
    b_next = copy.deepcopy(b_next_temp2)
    aot = copy.deepcopy(aot_temp2)
    os_.next(option, obs_backup)

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


# option的顺序
optionOrder_global = []
# 已经用过的option
option_use = []

if __name__ == '__main__':

    for r in range(1,11):
        # 记录50次的结果
        for i in range(20):
            # 写入的文件目录
            file_path = '../experiment_NO2G/experiment_iter/look_sys_data20_iter' + str(r * 2) + '.txt'

            # hor = 100
            hor = 3
            itr = r * 2

            VArr = [0 for i in range(hor + 1)]  # VArr[t]
            NArr = [0 for i in range(hor + 1)]  # NArr[t]
            dftDim = 18

            fileHeader = '../Synthetic_data_mc_20/'

            exps_N_Filename = fileHeader + 'aoTree20.txt'
            historyFilename = fileHeader + 'sys_data_history20.txt'

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

            # print(optionOrder)
            optionOrder_global = optionOrder
            os_ = OptionSampler(exps, optionOrder)

            aot = aoTree(exps, numOptions)
            # aot.reset()
            # print('bb cost=',aot.root.cost,'prob',aot.root.prob)

            # runtime0 = time.time()
            VArr, NArr = MC_method(b0, os_, aot, clfArr, hor, itr, VArr, NArr)
            # print("Program execution time:", time.time() - runtime0)
            # print('VArr =', VArr)
            # print("VArr len:", len(VArr))

            # 找到候选答案中的最大值
            content = str(VArr[len(VArr) - 1])  # 要写入的内容
            # 写入文件
            write_to_file(file_path, content)
