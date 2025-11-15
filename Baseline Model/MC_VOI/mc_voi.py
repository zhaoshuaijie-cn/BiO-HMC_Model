import random
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import time
import warnings
warnings.filterwarnings('ignore')
import copy

from option_sampler import OptionSampler
from ao_tree import aoTree

class BeliefState:
	__slots__ = ['t', 'history', 'belief', 'children', 'N', 'Vnc', 'Vc', 'V']

	def __init__(self, t):
		self.t = t
		self.history = None
		self.belief = None
		self.children = []
		self.N = 0
		self.Vnc = 0
		self.Vc = 0
		self.V = 0

	def addChild(self, b_next):
		for bc in self.children:
			# 'not (b_next.belief - bc.belief).any()' means 'b_next.belief == bc.belief'
			if not (b_next.belief - bc.belief).any():
				return bc
		self.children.append(b_next)
		return b_next
	
def sampleBeliefObservation(option, clfArr, b, b_next):
	b_next.history = copy.deepcopy(b.history)
	b_next.history[option].pop(0)

	# obs = int(clfArr[option].predict([b.history[option]])[0])
	# b_next.history[option].append(obs) # obs = 0:'-' / 1:'+'
	# return obs

	pr_ML_PosiObs = clfArr[option].predict_proba([b.history[option]])[0]
	if clfArr[option].classes_[0]: # classes_: [0 1] or [0] or [1]
		pr_ML_PosiObs = pr_ML_PosiObs[0]
	else:
		pr_ML_PosiObs = 1 - pr_ML_PosiObs[0]
	# pr_ML_PosiObs = np.isnan(pr_ML_PosiObs) and 1 or pr_ML_PosiObs
	rand = random.random()
	if pr_ML_PosiObs <= rand:
		b_next.history[option].append(0) # represent '-' observation
		return 0
	b_next.history[option].append(1) # represent '+' observation
	return 1

def sampleBeliefState(b, os, aot, clfArr, t):
	# option = random.randint(0, aot.numOptions-1) # or comment os

	option = os.sample()
	b_next = BeliefState(t)
	obs = sampleBeliefObservation(option, clfArr, b, b_next)
	aot.update(option, obs)
	os.next(option, obs)
	# print('option=',option,'obs=',obs)
	# print('cost=',aot.root.cost,'prob',aot.root.prob)

	b_next.belief = b.belief.copy()
	for i, node in enumerate(aot.leafArr):
		b_next.belief[i] = node.prob
	b_next.Vnc = aot.root.prob
	b_next = b.addChild(b_next)

	return (b_next, aot)


def calculateV(b, aot, C):
	b.N += 1
	# b.Vnc = (b.Vnc * (b.N-1) + Rnc) / b.N
	b_Vc = 0
	if len(b.children):
		for tempb in b.children:
			b_Vc += tempb.V * tempb.N
		b_Vc = b_Vc / b.N
		b.Vc = b_Vc - C
	b.V = (b.Vnc >= b.Vc) and b.Vnc or b.Vc

def sampleTrajectory(b, os, aot, clfArr, t, hor, C):
	if t < hor:
		t += 1
		b_next, aot = sampleBeliefState(b, os, aot, clfArr, t)
		sampleTrajectory(b_next, os, aot, clfArr, t, hor, C)
	calculateV(b, aot, C)

def MC_method(b0, os, aot, clfArr, hor, itr, C):
	while itr:
		os.reset()
		aot.reset()
		b0.Vnc = aot.root.prob
		sampleTrajectory(b0, os, aot, clfArr, 0, hor, C)
		print("iteration =",itr,"b0.V =",b0.V)
		itr -= 1

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

def recurPreTraversal(tCount_Arr, b):
	if b.Vnc >= b.Vc:
		# tCount_Arr[b.t] += b.N # print(sum(tCount_Arr)) # = itr
		tCount_Arr[b.t] += b.N * b.t
	elif len(b.children):
		for bc in b.children:
			recurPreTraversal(tCount_Arr, bc)

if __name__ == '__main__':
	# runtime0 = time.time()

	hor = 40
	itr = 2
	VArr = [0 for i in range(hor+1)] # VArr[t]
	NArr = [0 for i in range(hor+1)] # NArr[t]
	dftDim = 18

	fileHeader = '../Synthetic_data_mc_20/'

	exps_N_Filename = fileHeader + 'aoTree1.txt'
	historyFilename = fileHeader + 'sys_data_history1.txt'

	exps_N_File = open(exps_N_Filename)
	tmpArr = []
	for line in exps_N_File:
		tmpArr.append(line)
	exps = tmpArr[0]
	numOptions = int(tmpArr[1])
	optionOrder = list(map(int, tmpArr[2][:-1].split(',') ))
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
	b0.belief = np.ones(numOptions) * 0.5
	clfArr = [None] * numOptions
	for option in optionOrder:
		b0.history[option] = tmpArr[option][0]
		clfArr[option] = tmpArr[option][1]

	os = OptionSampler(exps, optionOrder)
	aot = aoTree(exps, numOptions)
	# aot.reset()
	# print('bb cost=',aot.root.cost,'prob',aot.root.prob)

	# 在某个固定工人成本的情况下，预期收益
	# C = 1e-3
	C = 0
	runtime0 = time.time()
	MC_method(b0, os, aot, clfArr, hor, itr, C)
	# print("MC-VOI execution time:",time.time() - runtime0)
	print('b0.V =', b0.V)

	tCount_Arr = np.zeros(hor+1)
	recurPreTraversal(tCount_Arr, b0)
	# print(tCount_Arr)
	# print(sum(tCount_Arr))
	# print('ave t =', sum(tCount_Arr)/itr)

	# ## [1] ########
	# C_Arr = [1e-2, 0.7e-2, 0.4e-2, \
	# 		1e-3, 0.7e-3, 0.4e-3, \
	# 		1e-4, 0.7e-4, 0.4e-4]
	# b0V_Arr = []
	# for C in C_Arr:
	# 	b0 = BeliefState(0)
	# 	b0.history = [None] * numOptions
	# 	b0.belief = np.ones(numOptions) * 0.5
	# 	clfArr = [None] * numOptions
	# 	for option in optionOrder:
	# 		b0.history[option] = tmpArr[option][0]
	# 		clfArr[option] = tmpArr[option][1]

	# 		os = OptionSampler(exps, optionOrder)
	# 		aot = aoTree(exps, numOptions)

	# 	MC_method(b0, os, aot, clfArr, hor, itr, C)
	# 	b0V_Arr.append(b0.V)
	# print(b0V_Arr)