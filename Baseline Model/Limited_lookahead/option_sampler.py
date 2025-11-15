from pyeda.inter import *

class OptionSampler:
	__slots__ = ['sat_all', 'optionOrder', 'curr_sa_idxArr', 'count', 'countFlag', 'NOptionsArr']

	def __init__(self, exps, optionOrder):
		self.count = 0
		self.countFlag = True
		self.NOptionsArr = []

		ch = None
		bl_expr = []
		d = ''
		for i in range(len(exps)-2,-1,-1): # 'len(exps)-2' because exps[-1] = '\n'
			if exps[i].isdigit():
				d = exps[i] + d
			else:
				if d:
					bl_expr.append('c'+d)
					d = ''
				bl_expr.append(exps[i])
		if d:
			bl_expr.append('c'+d)
		bl_expr.reverse()
		bl_expr = expr(''.join(bl_expr))

		# order sat_all acc. to optionOrder
		sat_all = list(bl_expr.satisfy_all())
		tmpArr = []
		for sat_one in sat_all:
			tmpArr.append([])
			for o in optionOrder:
				if exprvar('c'+str(o)) in sat_one:
					tmpArr[-1].append((o, sat_one[exprvar('c'+str(o))]))

		tmpArr.append([])
		for o in optionOrder:
			tmpArr[-1].append((o, 'any'))

		self.sat_all = tmpArr

		self.optionOrder = optionOrder
		self.curr_sa_idxArr = [0] * len(self.sat_all) # 'sa': sat_all

	def sample(self):
		optionOrder = self.optionOrder
		idxes = []
		for sat_one,idx in zip(self.sat_all,self.curr_sa_idxArr):
			if -1 != idx: # mark
				idxes.append(optionOrder.index(sat_one[idx][0]))
		return optionOrder[min(idxes)]

	def next(self, option, obs):
		curr_sa_idxArr = self.curr_sa_idxArr
		idx = None
		for i,sat_one in enumerate(self.sat_all):
			idx = curr_sa_idxArr[i]
			if -1 == idx:
				continue
			if sat_one[idx][0] == option:
				if sat_one[idx][1] == obs or sat_one[idx][1] == 'any':
					idx += 1
					if idx == len(sat_one):
						curr_sa_idxArr[i] = -1

						if -1 == curr_sa_idxArr[-1]:
							self.curr_sa_idxArr = [0] * len(self.sat_all)
							if self.countFlag:
								self.count += 1
								self.NOptionsArr.append(len(self.optionOrder))
							self.countFlag = True
						else:
							self.count += 1
							self.NOptionsArr.append(self.optionOrder.index(option)+1)
							self.countFlag = False
					else:
						curr_sa_idxArr[i] = idx
				else:
					curr_sa_idxArr[i] = -1
		
		# if sum(curr_sa_idxArr) == -1 * len(self.sat_all): # mark
		# 	self.curr_sa_idxArr = [0] * len(self.sat_all)

	def reset(self):
		self.curr_sa_idxArr = [0] * len(self.sat_all)