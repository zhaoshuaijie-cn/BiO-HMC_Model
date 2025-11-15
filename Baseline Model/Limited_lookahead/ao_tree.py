class Node:
	__slots__ = ['operator', 'parent', 'children', 'prob', 'cost']

	def __init__(self, op):
		self.operator = op
		self.parent = None

		# for not leaf node
		self.children = []

		# for init leaf node (op = 'l')
		self.prob = 0.5
		self.cost = 1

	def compute(self):
		arr = None
		children = self.children
		if '&' == self.operator:
			arr = [(1 - child.prob) / child.cost for child in children]
		else:
			arr = [child.prob / child.cost for child in children]

		srt = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
		prob = 1
		cost = 0
		sumNP = 1  # 'NP': negative prob

		if '&' == self.operator:
			for i, _ in srt:
				cost += children[i].cost * sumNP
				sumNP *= children[i].prob

			for child in children:
				prob *= child.prob
		else:
			for i, _ in srt:
				cost += children[i].cost * sumNP
				sumNP *= 1 - children[i].prob

			for child in children:
				prob *= 1 - child.prob
			prob = 1 - prob

		self.prob = prob
		self.cost = cost

	def reset_prob_cost(self):
		self.prob = 0.5
		self.cost = 1


class aoTree:
	__slots__ = ['root', 'numOptions', 'leafArr']

	def __cal(self, NStack, op):
		node2 = NStack.pop()
		node1 = NStack.pop()

		# condition 'op == node1.operator and op == node2.operator' doesn't exist
		if op != node1.operator and op != node2.operator: # '!=' before 'and'
			node = Node(op)
			node.children.append(node1)
			node.children.append(node2)
			# node.compute()
			node1.parent = node
			node2.parent = node
			NStack.append(node)
		elif op == node1.operator:
			node1.children.append(node2)
			# node1.compute()
			node2.parent = node1
			NStack.append(node1)
		else:
			node2.children.append(node1)
			# node2.compute()
			node1.parent = node2
			NStack.append(node2)

		return NStack

	# 'exps': expression
	def __init__(self, exps, numOptions):
		NStack = []
		OStack = []
		s = []
		number = None
		self.numOptions = numOptions
		self.leafArr = [None] * numOptions

		for ch in exps:
			if ch not in ['&','|','(',')','\n']:
				s.append(ch)
				continue
			if len(s) > 0:
				NStack.append(Node('l'))
				number = int(''.join(s))
				self.leafArr[number] = NStack[-1]
				s.clear()

			if '&' == ch:
				while len(OStack) > 0:
					ch0 = OStack[-1]
					if '&' == ch0:
						OStack.pop()
						NStack = self.__cal(NStack, ch0)
					else:
						break
				OStack.append(ch)
			elif '|' == ch:
				while len(OStack) > 0:
					ch0 = OStack[-1]
					if '&' == ch0 or '|' == ch0:
						OStack.pop()
						NStack = self.__cal(NStack, ch0)
					else:
						break
				OStack.append(ch)
			elif '(' == ch:
				OStack.append(ch)
			elif ')' == ch:
				ch0 = OStack.pop()
				while '(' != ch0:
					NStack = self.__cal(NStack, ch0)
					ch0 = OStack.pop()
			else:
				while OStack:
					ch0 = OStack.pop()
					NStack = self.__cal(NStack, ch0)
		self.root = NStack.pop()

	def __postOrder(self, node):
		if node.children: # not a leaf node
			for child in node.children:
				self.__postOrder(child)
			node.compute()
		else:
			node.reset_prob_cost()

	# reset prob & cost of all nodes
	def reset(self):
		self.__postOrder(self.root)

	def update(self, option, obs):
		node = self.leafArr[option]

		posiP = 0.75
		unNormP = None
		if obs:
			unNormP = node.prob * posiP
			node.prob = unNormP / (unNormP + \
				(1 - node.prob) * (1 - posiP))
		else:
			unNormP = node.prob * (1 - posiP)
			node.prob = unNormP / (unNormP + \
				(1 - node.prob) * posiP)

		while node != self.root:
			node = node.parent
			node.compute()