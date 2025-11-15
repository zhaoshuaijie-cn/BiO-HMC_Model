import copy
import itertools
import pandas as pd
import numpy as np
import random
import time
import os
from collections import OrderedDict

# 记录程序开始时间
start_time = time.time()

# 自定义一个异常类
class BreakLoop(Exception):
    pass

# 生成幂集，并处理，返回一个列表
def generate_power_set(set_o):
    power_set_o = list(
        itertools.chain.from_iterable(itertools.combinations(set_o, r)
                                      for r in range(len(set_o) + 1)))

    for i in range(len(power_set_o)):
        power_set_o[i] = list(power_set_o[i])

    return power_set_o


# 公式14 U_i的计算
def formula_u_compute(set_o_part,set_o_neg,c_i):
    prob_ = 1
    set_o_union = set_o_part + set_o_neg
    # print("set_o_union",set_o_union)

    for i in range( len(set_o_union) ):
        prob_ = prob_ * (1 - cond_prob_table.at[set_o_union[i],'c'+str(c_i)])

    # print("prob_ before:",prob_)
    prob_ = prob_ * candidate_prob[c_i-1] + 1 - candidate_prob[c_i-1]
    # print("prob_ after:",prob_)
    return prob_


# 公式13重写后的式子的计算，其中
def formula_completion(set_o_pos,set_o_neg):
    # 生成集合o的幂集
    power_set_o = generate_power_set(set_o_pos)

    sum_ = 0
    for i in range(len(power_set_o)):
        temp = pow( -1,len(power_set_o[i]) )
        for j in range(candidate_num):
            temp = temp * formula_u_compute(power_set_o[i],set_o_neg,j+1)
        # print("temp:",temp)
        sum_ = sum_ + temp

    return sum_

# 梯度的计算 分为两个
def candiate_gradient_compute(set_o_pos,set_o_neg,c_i):
    # 生成幂集
    power_set_o = generate_power_set(set_o_pos)
    # print(power_set_o)

    sum_ = 0
    for i in range( len(power_set_o) ):
        temp = pow( -1, len(power_set_o[i]) )
        for j in range(1,candidate_num+1):
            if j == c_i: continue
            temp = temp * formula_u_compute(power_set_o[i],set_o_neg,j)

        temp1 = 1
        power_set_union = power_set_o[i]+set_o_neg
        for k in range( len(power_set_union) ):
            temp1 = temp1 * (1 - cond_prob_table.at[power_set_union[k],'c'+str(c_i)])
        temp1 = temp1 - 1
        # print("temp1:",temp1)

        temp = temp * temp1
        sum_ = sum_ + temp

    return sum_


def option_cond_gradient_compute(set_o_pos,set_o_neg,o_i,c_i):
    o_i_str = 'o'+str(o_i)
    set_o_pos_remove = set_o_pos[:]
    if o_i_str in set_o_pos_remove:
        set_o_pos_remove.remove(o_i_str)

    # 生成幂集
    power_set_o_pos = generate_power_set(set_o_pos_remove)
    # print(power_set_o_pos)

    sum_ = 0
    for i in range( len(power_set_o_pos) ):
        temp = pow( -1,len(power_set_o_pos[i])+1 )

        temp_u = 1
        for j in range(1,candidate_num+1):
            if j == c_i: continue
            new_set_part = power_set_o_pos[i] + [o_i_str]
            # print("new_set_part",new_set_part)
            temp_u = temp_u * formula_u_compute(new_set_part,set_o_neg,j)

        temp1 = 1
        new_set_part1 = power_set_o_pos[i] + set_o_neg
        if o_i_str in new_set_part1:
            new_set_part1.remove(o_i_str)

        for k in range( len(new_set_part1) ):
            temp1 = temp1 * (1 - cond_prob_table.at[new_set_part1[k],'c'+str(c_i)]) * candidate_prob[c_i-1]

        # print("sum_:",sum_," temp:",temp," temp_u:",temp_u," temp1:",temp1)
        sum_ = sum_ + temp * temp_u * temp1

    return sum_


# 公式14 U_i的计算，与上面参数不同
def formula_u_compute_param(candidate_prob_,cond_prob_table_,set_o_part,set_o_neg,c_i):
    prob_ = 1
    set_o_union = set_o_part + set_o_neg
    # print("set_o_union",set_o_union)

    for i in range( len(set_o_union) ):
        prob_ = prob_ * (1 - cond_prob_table_.at[set_o_union[i],'c'+str(c_i)])

    # print("prob_ before:",prob_)
    prob_ = prob_ * candidate_prob_[c_i-1] + 1 - candidate_prob_[c_i-1]
    # print("prob_ after:",prob_)
    return prob_

def formula_completion_param(candidate_prob_,cond_prob_table_,set_o_pos,set_o_neg):
    # 生成集合o的幂集
    power_set_o = generate_power_set(set_o_pos)

    sum_ = 0
    # 遍历生成的幂集，对每个元素进行计算
    for i in range(len(power_set_o)):
        temp = pow( -1,len(power_set_o[i]) )
        for j in range(candidate_num):
            temp = temp * formula_u_compute_param(candidate_prob_,cond_prob_table_,power_set_o[i],set_o_neg,j+1)
        # print("temp:",temp)
        sum_ = sum_ + temp

    return sum_

def projected_gradient_method(param,l_bound,r_bound):
    if l_bound < param < r_bound:
        return param
    elif l_bound > param:
        return l_bound
    elif r_bound < param:
        return r_bound

# 参数有两种，最后的更新也分别处理
def parameter_update_candidate(diff,set_o_prob,c_i,gamma_,set_o_pos,set_o_neg):
    sigma = 0.01
    beta = 0.9
    l_bound = 0.005
    r_bound = 0.990

    candidate_prob_temp = candidate_prob[:]
    cond_prob_table_temp = cond_prob_table[:]
    # 原参数值
    x = candidate_prob[c_i-1]
    grad_ = diff / set_o_prob

    x_new = x + gamma_ * grad_
    x_old = x

    candidate_prob_temp[c_i-1] = x_new
    gamma_new = gamma_

    iterations = 10

    if formula_completion_param(candidate_prob_temp,cond_prob_table_temp,set_o_pos,set_o_neg) - set_o_prob >= sigma * grad_ * (x_new - x):
        while iterations:
            gamma_ = gamma_new
            gamma_new = gamma_new / beta
            x_new = projected_gradient_method(x + gamma_new * grad_,l_bound,r_bound)

            candidate_prob_temp[c_i-1] = x_new
            print("x =", x, "x_new =", x_new)
            print("f(x_new) =",formula_completion_param(candidate_prob_temp,cond_prob_table_temp,set_o_pos,set_o_neg),"f(x)",set_o_prob)
            print("gamma_new =",gamma_new)

            if formula_completion_param(candidate_prob_temp,cond_prob_table_temp,set_o_pos,set_o_neg) - \
                    set_o_prob >= sigma * grad_ * (x_new - x) and x_new != x_old:

                x_old = x_new
                iterations -= 1
                continue
            else:
                break
    else: # 不满足条件
        while iterations:
            gamma_new = gamma_new * beta
            x_new = x + gamma_new * grad_
            candidate_prob_temp[c_i - 1] = x_new
            if formula_completion_param(candidate_prob_temp,cond_prob_table_temp,set_o_pos,set_o_neg) - \
                    set_o_prob >= sigma * grad_ * (x_new - x) or x_new == x_old:

                gamma_ = gamma_new
                break
            else:
                iterations -= 1
                x_old = x_new
                continue

    param_new = projected_gradient_method(x + gamma_ * grad_,l_bound,r_bound)
    return param_new,gamma_

def parameter_update_option_cond_prob():
    return 1

def armijo_rule_eta(param_old, set_o_prob, diff, flag):
    beta = 0.1
    sigma = 0.01
    grad_ = diff / set_o_prob
    eta = 1
    param_new = param_old + eta * grad_

    set_o_positive = []
    set_o_negitive = []

    for i in range(len(candidate_prob)):
        index = i+1
        worker_answer = random.choice(history_data[index - 1])
        if worker_answer == 1:
            set_o_positive.append('o' + str(index))
        else:
            set_o_negitive.append('o' + str(index))

    set_o_prob_new = formula_completion(set_o_positive, set_o_negitive)
    # print(set_o_prob_new,set_o_prob)
    k = 0
    while set_o_prob_new <= set_o_prob + sigma * grad_ * (param_new-param_old):
        # print(set_o_prob_new, set_o_prob)
        eta = eta * beta
        param_new = param_new + eta * grad_
        k += 1
        if k >= 5:
            break
        # print(eta)

    l_bound = 0.0005
    r_bound = 0.99

    param_new = projected_gradient_method(param_old + eta * grad_, l_bound, r_bound)
    return param_new

# 参数更新
def parameter_update(param_old,set_o_prob,diff,flag):
    grad_ = diff / set_o_prob

    gamma = 0.01
    l_bound = 0.0005
    r_bound = 0.99

    if flag == 0:
        gamma = 0.01
    else:
        gamma = 0.02

    param_new = projected_gradient_method(param_old + gamma * grad_,l_bound,r_bound)
    return param_new


gamma_cand = 1.0
gamma_op = 1.0

# 计算对一个志愿者提问以后，对于两个参数的更新
def each_worker_option_update(set_o_posi,set_o_neg,o_i,c_i):
    set_o_prob = formula_completion(set_o_posi, set_o_neg)

    diff1 = candiate_gradient_compute(set_o_posi, set_o_neg, c_i)
    diff2 = option_cond_gradient_compute(set_o_posi, set_o_neg, o_i, c_i)

    global gamma_cand
    res1 = parameter_update(candidate_prob[c_i-1], set_o_prob, diff1,0)
    # res1, gamma_cand = parameter_update_candidate(diff1, set_o_prob, c_i, gamma_cand, set_o_posi, set_o_neg)
    candidate_prob[c_i - 1] = res1

    res2 = 1 - parameter_update(1 - cond_prob_table.at['o'+str(o_i), 'c'+str(c_i)], set_o_prob, diff2,1)
    cond_prob_table.at['o'+str(o_i), 'c'+str(c_i)] = res2

test_set_pos = ['o1']
test_set_neg = []

# 测试代码
def test():
    print("Test Start...")
    for i in range(9):
        # print("test_set_pos",test_set_pos)
        set_o_prob = formula_completion(test_set_pos,test_set_neg)
        # print("set_o_prob",set_o_prob)
        diff1 = candiate_gradient_compute(test_set_pos,test_set_neg,2)
        # print("diff1:",diff1)
        diff2 = option_cond_gradient_compute(test_set_pos,test_set_neg,3,2)
        # print("diff2:",diff2)

        res1 = parameter_update(candidate_prob[1],set_o_prob,diff1)
        candidate_prob[1] = res1
        print(candidate_prob)

        res2 = 1 - parameter_update(1 - cond_prob_table.at['o3','c2'],set_o_prob,diff2)
        cond_prob_table.at['o3','c2'] = res2
        print(cond_prob_table)
        print()
    print("Test End...")

# 数据集
fileHeader = '../Synthetic_data/'

data_Filename = fileHeader + 'sys_data20.txt'
history_Filename = fileHeader + 'sys_data_history20.txt'

cand_tuple = set()
op_tuple = set()
# 处理图结构数据
with open(data_Filename, 'r') as file:
    for line in file:
        # 使用 strip() 去除行尾的换行符
        # print(line.strip())

        numbers_str = line.split()
        numbers = [int(num) for num in numbers_str]
        # print(numbers)

        cand_tuple.add(numbers[0])
        op_tuple.add(numbers[1])

# print(cand_tuple,op_tuple)
# 创建表格，其中元素均初始化为0
cand_list = []
op_list = []

option_dict = {}
candidate_dict = {}
for i in range( 1,len(cand_tuple)+1 ):
    cand_list.append('c'+str(i))
    candidate_dict['c'+str(i)] = set()

for j in range( 1,len(op_tuple)+1 ):
    op_list.append('o'+str(j))
    option_dict['o'+str(j)] = set()
# print(cand_list,op_list)
# print(option_dict)
# print(candidate_dict)

# 指定列名和行名
columns = cand_list
index = op_list

zero_data = np.zeros((len(index), len(columns)))
# 创建一个空的 DataFrame
cond_prob_table = pd.DataFrame(zero_data,index=index, columns=columns)
# print(cond_prob_table)

candidate_prob = np.full(len(columns),1/len(columns))
# print(candidate_prob)
candidate_num = len(candidate_prob)

with open(data_Filename, 'r') as file:
    for line in file:
        # 使用 strip() 去除行尾的换行符
        # print(line.strip())
        numbers_str = line.split()
        numbers = [int(num) for num in numbers_str]

        option_dict['o'+str(numbers[1])].add('c'+str(numbers[0]))
        candidate_dict['c'+str(numbers[0])].add('o'+str(numbers[1]))

        cond_prob_table.at[ 'o'+str(numbers[1]), 'c'+str(numbers[0]) ]=0.5

# print(cond_prob_table)
# print("option_dict:",option_dict)
# print("candidate_dict =",candidate_dict)

# 读入历史观测数据
history_data = []
history_data_num = 0
with open(history_Filename, 'r') as file:
    for line in file:
        numbers_str = line.split()
        numbers = [int(num) for num in numbers_str]
        history_data_num = len(numbers)
        history_data.append(numbers)

# print(history_data)
# print(history_data_num)

# 按照自然顺序为提问顺序，对于每个问题，从其对应的历史观测数据中 循环进行采样（即按照历史观测数据中的顺序依次作为答案
def sample_option_circulation():
    print("Sample option circulation...")

    for s in range(history_data_num):
        for t in range( len(op_list) ):
            set_o_positive = []
            set_o_negitive = []
            if history_data[t][s] == 1:
                set_o_positive.append('o'+str(t+1))
            else:
                set_o_negitive.append('o'+str(t+1))

            # print("history_number:",s,'-',t)
            for key in option_dict:
                if key == 'o'+str(t+1):
                    for element in option_dict[key]:
                        cand_seq_ = int( element[1:] )
                        # 找到该选项对应的候选后 再去更新
                        each_worker_option_update(set_o_positive,set_o_negitive,t+1,cand_seq_)
                    break


# 对历史观测数据不同的采样想法，这个random指的是从历史数据随机采样一个
# 按照自然顺序为提问顺序，对于每个问题，从其对应的历史观测数据中随机采样一个 作为当前志愿者的回答
def sample_option_random(iteration):
    # print("Sample random...")

    try:
        # 迭代iteration次
        for s in range(iteration):
            # print("Iteration -",s+1)

            for t in range( len(op_list) ):
                worker_answer = random.choice(history_data[t])

                set_o_positive = []
                set_o_negitive = []
                if worker_answer == 1:
                    set_o_positive.append('o' + str(t + 1))
                else:
                    set_o_negitive.append('o' + str(t + 1))

                # 更新与该选项有边的候选的概率、
                for key in option_dict:
                    if key == 'o' + str(t + 1):
                        for element in option_dict[key]:
                            cand_seq_ = int(element[1:])
                            # 找到该选项对应的候选后 再去更新
                            each_worker_option_update(set_o_positive, set_o_negitive, t + 1, cand_seq_)

                            for u in range(len(candidate_prob)):
                                if abs(candidate_prob[u] - 1) <= 0.02 :
                                    # print("Exist a candidate closed to 1...")
                                    raise BreakLoop  # 当满足条件时，抛出自定义异常退出所有循环
                                elif candidate_prob[u] <= 0:
                                    # print("Exist a candidate less than 0...")
                                    raise BreakLoop
                        break
        # print()

    except BreakLoop:
        print("Exited all loops")
        print("Iteration",s+1, "times ending...")

    # print()

# 按照打乱的提问顺序进行提问
def sample_option_random_seq(iteration,option_list):
    # print("Sample random...")

    try:
        # 迭代iteration次
        for s in range(iteration):
            # print("Iteration -",s+1)
            random.shuffle(option_list)

            for t in range( len(option_list) ):
                question_seq = int(option_list[t][1:]) - 1

                worker_answer = random.choice(history_data[question_seq])

                # 得到对option的回答以后更新两个参数
                set_o_positive = []
                set_o_negitive = []
                if worker_answer == 1:
                    set_o_positive.append('o' + str(question_seq + 1))
                else:
                    set_o_negitive.append('o' + str(question_seq + 1))

                for key in option_dict:
                    if key == 'o' + str(question_seq + 1):
                        for element in option_dict[key]:
                            cand_seq_ = int(element[1:])
                            # 找到该选项对应的候选后 再去更新
                            each_worker_option_update(set_o_positive, set_o_negitive, question_seq + 1, cand_seq_)

                            for u in range(len(candidate_prob)):
                                if abs(candidate_prob[u] - 1) <= 0.02 :
                                    # print("Exist a candidate closed to 1...")
                                    raise BreakLoop  # 当满足条件时，抛出自定义异常退出所有循环
                                elif candidate_prob[u] <= 0:
                                    # print("Exist a candidate less than 0...")
                                    raise BreakLoop
                        break
        # print()

    except BreakLoop:
        print("Exited all loops")
        print("Iteration",s+1, "times ending...")
    # print()

# 添加元素的函数
def add_to_ordered_set(value,my_ordered_set):
    if value not in my_ordered_set:
        my_ordered_set[value] = None

# 按照指定的顺序进行提问，并且每次迭代之后都会改变顺序
def sample_option_random_order(iteration, option_list):
    # print("Sample random...")

    try:
        # 迭代iteration次
        for s in range(iteration):
            print("Iteration -",s+1)

            # 查看每次的提问顺序
            print("question order =",option_list)

            for t in range(len(option_list)):
                question_seq = int(option_list[t][1:]) - 1

                worker_answer = random.choice(history_data[question_seq])

                set_o_positive = []
                set_o_negitive = []
                if worker_answer == 1:
                    set_o_positive.append('o' + str(question_seq + 1))
                else:
                    set_o_negitive.append('o' + str(question_seq + 1))

                for key in option_dict:
                    if key == 'o' + str(question_seq + 1):
                        for element in option_dict[key]:
                            cand_seq_ = int(element[1:])
                            each_worker_option_update(set_o_positive, set_o_negitive, question_seq + 1, cand_seq_)

                            for u in range(len(candidate_prob)):
                                if abs(candidate_prob[u] - 1) <= 0.02:
                                    # print("Exist a candidate closed to 1...")
                                    raise BreakLoop  # 当满足条件时，抛出自定义异常退出所有循环
                                elif candidate_prob[u] <= 0:
                                    # print("Exist a candidate less than 0...")
                                    raise BreakLoop
                        break

            candidate_prob_temp = copy.deepcopy(candidate_prob)

            indexed_list = list(enumerate(candidate_prob_temp))

            sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

            my_ordered_set = OrderedDict()
            for original_index, value in sorted_list:
                for key in candidate_dict:
                    if key == 'c' + str(original_index + 1):
                        for element in candidate_dict[key]:
                            add_to_ordered_set(element,my_ordered_set)

            option_list = list(my_ordered_set.keys())
            # print("new question order =",option_list)

        # print()

    except BreakLoop:
        print("Exited all loops")
        print("Iteration", s + 1, "times ending...")

    # print()

# 上面的方法只对候选答案排序了，没有对同一个候选答案对应的几个选项排序，这里再加上对选项的排序
def sample_option_random_order_all(iteration, option_list):
    # print("Sample random...")

    try:
        # 迭代iteration次
        for s in range(iteration):
            # print("Iteration -",s+1)

            # 查看每次的提问顺序
            # print("question order =",option_list)

            for t in range(len(option_list)):
                question_seq = int(option_list[t][1:]) - 1

                worker_answer = random.choice(history_data[question_seq])
                # 得到对option的回答以后更新两个参数
                set_o_positive = []
                set_o_negitive = []
                if worker_answer == 1:
                    set_o_positive.append('o' + str(question_seq + 1))
                else:
                    set_o_negitive.append('o' + str(question_seq + 1))

                for key in option_dict:
                    if key == 'o' + str(question_seq + 1):
                        for element in option_dict[key]:
                            cand_seq_ = int(element[1:])

                            each_worker_option_update(set_o_positive, set_o_negitive, question_seq + 1, cand_seq_)

                            for u in range(len(candidate_prob)):
                                if abs(candidate_prob[u] - 1) <= 0.02:
                                    # print("Exist a candidate closed to 1...")
                                    raise BreakLoop  # 当满足条件时，抛出自定义异常退出所有循环
                                elif candidate_prob[u] <= 0:
                                    # print("Exist a candidate less than 0...")
                                    raise BreakLoop
                        break

            candidate_prob_temp = copy.deepcopy(candidate_prob)
            indexed_list = list(enumerate(candidate_prob_temp))
            sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

            my_ordered_set = OrderedDict()
            for original_index, value in sorted_list:
                for key in candidate_dict:
                    if key == 'c' + str(original_index + 1):
                        key_option_list = []
                        for element in candidate_dict[key]:
                            key_option_list.append(element)

                        for x in range(len(key_option_list)):
                            for y in range(x + 1,len(key_option_list)):
                                if cond_prob_table.at[key_option_list[x],key] >= cond_prob_table.at[key_option_list[y],key]:
                                    continue
                                else:
                                    tmp_ = key_option_list[x]
                                    key_option_list[x] = key_option_list[y]
                                    key_option_list[y] = tmp_

                        for z in range(len(key_option_list)):
                            add_to_ordered_set(key_option_list[z],my_ordered_set)

            option_list = list(my_ordered_set.keys())
            # option_list.reverse()
            # print("new question order =",option_list)

        # print()

    except BreakLoop:
        print("Exited all loops")
        print("Iteration", s + 1, "times ending...")

    # print()


# 按照P(c)的后验概率进行排序
def sample_option_random_order_posterior(iteration, option_list):
    # print("Sample random...")

    try:
        # 迭代iteration次
        for s in range(iteration):
            # print("Iteration -",s+1)

            # 查看每次的提问顺序
            # print("question order =",option_list)

            for t in range(len(option_list)):
                question_seq = int(option_list[t][1:]) - 1

                worker_answer = random.choice(history_data[question_seq])

                set_o_positive = []
                set_o_negitive = []
                if worker_answer == 1:
                    set_o_positive.append('o' + str(question_seq + 1))
                else:
                    set_o_negitive.append('o' + str(question_seq + 1))

                for key in option_dict:
                    if key == 'o' + str(question_seq + 1):
                        for element in option_dict[key]:
                            cand_seq_ = int(element[1:])
                            each_worker_option_update(set_o_positive, set_o_negitive, question_seq + 1, cand_seq_)

                            for u in range(len(candidate_prob)):
                                if abs(candidate_prob[u] - 1) <= 0.02:
                                    # print("Exist a candidate closed to 1...")
                                    raise BreakLoop
                                elif candidate_prob[u] <= 0:
                                    # print("Exist a candidate less than 0...")
                                    raise BreakLoop
                        break

            candidate_prob_temp = copy.deepcopy(candidate_prob)
            indexed_list = list(enumerate(candidate_prob_temp))
            sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

            my_ordered_set = OrderedDict()
            for original_index, value in sorted_list:
                # print("original_index =",original_index,"value =",value)
                for key in candidate_dict:
                    if key == 'c' + str(original_index + 1):
                        key_option_list = []
                        for element in candidate_dict[key]:
                            key_option_list.append(element)

                        for x in range(len(key_option_list)):
                            for y in range(x + 1,len(key_option_list)):
                                if cond_prob_table.at[key_option_list[x],key] * candidate_prob[key] >= \
                                        cond_prob_table.at[key_option_list[y],key] * candidate_prob[key]:
                                    continue
                                else:
                                    tmp_ = key_option_list[x]
                                    key_option_list[x] = key_option_list[y]
                                    key_option_list[y] = tmp_

                        for z in range(len(key_option_list)):
                            add_to_ordered_set(key_option_list[z],my_ordered_set)

            option_list = list(my_ordered_set.keys())
            # print("new question order =",option_list)

        # print()

    except BreakLoop:
        print("Exited all loops")
        print("Iteration", s + 1, "times ending...")

    # print()


# 按照候选答案降序进行排序
def sample_option_random_low_cand(iteration, option_list):
    # print("Sample random...")

    try:
        # 迭代iteration次
        for s in range(iteration):
            # print("Iteration -",s+1)

            # 查看每次的提问顺序
            # print("question order =",option_list)

            for t in range(len(option_list)):
                question_seq = int(option_list[t][1:]) - 1

                worker_answer = random.choice(history_data[question_seq])

                set_o_positive = []
                set_o_negitive = []
                if worker_answer == 1:
                    set_o_positive.append('o' + str(question_seq + 1))
                else:
                    set_o_negitive.append('o' + str(question_seq + 1))

                for key in option_dict:
                    if key == 'o' + str(question_seq + 1):
                        for element in option_dict[key]:
                            cand_seq_ = int(element[1:])

                            each_worker_option_update(set_o_positive, set_o_negitive, question_seq + 1, cand_seq_)

                            for u in range(len(candidate_prob)):
                                if abs(candidate_prob[u] - 1) <= 0.02:
                                    # print("Exist a candidate closed to 1...")
                                    raise BreakLoop  # 当满足条件时，抛出自定义异常退出所有循环
                                elif candidate_prob[u] <= 0:
                                    # print("Exist a candidate less than 0...")
                                    raise BreakLoop
                        break

            candidate_prob_temp = copy.deepcopy(candidate_prob)
            indexed_list = list(enumerate(candidate_prob_temp))
            # sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
            sorted_list = sorted(indexed_list, key=lambda x: x[1])

            my_ordered_set = OrderedDict()
            for original_index, value in sorted_list:
                # print("original_index =",original_index,"value =",value)
                for key in candidate_dict:
                    if key == 'c' + str(original_index + 1):
                        key_option_list = []
                        for element in candidate_dict[key]:
                            key_option_list.append(element)

                        for x in range(len(key_option_list)):
                            for y in range(x + 1,len(key_option_list)):
                                if cond_prob_table.at[key_option_list[x],key] <= cond_prob_table.at[key_option_list[y],key]:
                                    continue
                                else:
                                    tmp_ = key_option_list[x]
                                    key_option_list[x] = key_option_list[y]
                                    key_option_list[y] = tmp_

                        # print("sorted =",key_option_list)

                        for z in range(len(key_option_list)):
                            add_to_ordered_set(key_option_list[z],my_ordered_set)

            option_list = list(my_ordered_set.keys())
            # option_list.reverse()
            # print("new question order =",option_list)

        # print()

    except BreakLoop:
        print("Exited all loops")
        print("Iteration", s + 1, "times ending...")

    # print()

# 记录程序结束时间
end_time = time.time()

# 计算执行时间
# execution_time = end_time - start_time
# print("Program execution time:",execution_time)

# 对于迭代次数的对比，分别记录迭代次数为5、10、15、20次情况下的结果
def write_to_file(file_path, content):
    # 确保目录存在
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'a') as file:
        file.write(content + '\n')  # 添加内容并换行


candidate_prob_backup = copy.deepcopy(candidate_prob[:])
cond_prob_table_backup = copy.deepcopy(cond_prob_table[:])

for r in range(1,11):
    # 记录50次的结果
    for i in range(20):
        option_list_order = copy.deepcopy(op_list)
        # print(option_list_order)

        op_list_rand = copy.deepcopy(op_list)
        # random.shuffle(op_list_rand)
        # print(op_list_rand)

        # 写入的文件目录
        # file_path = '../experiment_strategy/sys_data2_natural' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/sys_data20_random' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/sys_data2_order' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/sys_data2_order_all' + str(r * 2) + '.txt'

        # file_path = '../experiment_strategy/sys_data3_natural' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/sys_data3_random' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/sys_data3_order' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/sys_data3_order_all' + str(r * 2) + '.txt'

        # file_path = '../experiment_strategy/galaxyzoo_natural' + str(r*2) + '.txt'
        # file_path = '../experiment_strategy/galaxyzoo_random4_' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/galaxyzoo_order' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/galaxyzoo_order_all' + str(r * 2) + '.txt'
        # file_path = '../experiment_strategy/galaxyzoo_order_low1_' + str(r * 2) + '.txt'

        file_path = '../experiment_no2g/experiment_strategy/sys_data20_order_low' + str(r * 2) + '.txt'

        # print("can =",candidate_prob)

        # sample_random(5)

        # sample_option_random(r*2)

        # print("op_list_rand =", op_list_rand)
        # sample_option_random_seq(r*2, op_list_rand)

        # sample_option_random_order(r*2,option_list_order)

        # sample_option_random_order_all(r * 2, option_list_order)

        sample_option_random_low_cand(r * 2, option_list_order)


        # 展示更新后的结果
        # print("candidate prob: ", candidate_prob)
        # print("conditional probabilistic： ")
        # print(cond_prob_table)

        content = str(max(candidate_prob))  # 要写入的内容
        # 写入文件
        write_to_file(file_path, content)

        # print("cand_bakcup =",candidate_prob_backup)
        candidate_prob = copy.deepcopy(candidate_prob_backup[:])
        cond_prob_table = copy.deepcopy(cond_prob_table_backup[:])

# 展示更新后的结果
# print("candidate prob: ",candidate_prob)
# print("conditional probabilistic： ")
# print(cond_prob_table)