import copy
import itertools
import math

import pandas as pd
import numpy as np
import random
import time
import os

# 自定义一个异常类
class BreakLoop(Exception):
    pass

# 生成幂集，并处理，返回一个列表
def generate_power_set(set_o):
    power_set_o = list(
        itertools.chain.from_iterable(itertools.combinations(set_o, r)
                                      for r in range(len(set_o) + 1)))

    # 将其中的每个元素转化为列表
    for i in range(len(power_set_o)):
        power_set_o[i] = list(power_set_o[i])

    return power_set_o

# 公式14 U_i的计算
def formula_u_compute(set_o_part, set_o_neg, c_i):
    prob_ = 1
    set_o_union = set_o_part + set_o_neg
    # print("set_o_union",set_o_union)

    for i in range(len(set_o_union)):
        prob_ = prob_ * (1 - cond_prob_table.at[set_o_union[i], 'c' + str(c_i)])

    prob_ = prob_ * candidate_prob[c_i - 1] + 1 - candidate_prob[c_i - 1]
    return prob_

# 公式13重写后的式子的计算，其中
def formula_completion(set_o_pos, set_o_neg):
    # 生成集合o的幂集
    power_set_o = generate_power_set(set_o_pos)

    sum_ = 0
    # 遍历生成的幂集，对每个元素进行计算
    for i in range(len(power_set_o)):
        temp = pow(-1, len(power_set_o[i]))
        for j in range(candidate_num):
            # 下标从1开始，遍历的索引是从0开始的
            temp = temp * formula_u_compute(power_set_o[i], set_o_neg, j + 1)
        # print("temp:",temp)
        sum_ = sum_ + temp

    return sum_

# 梯度的计算 分为两个
def candiate_gradient_compute(set_o_pos, set_o_neg, c_i):
    # 生成幂集
    power_set_o = generate_power_set(set_o_pos)
    # print(power_set_o)

    sum_ = 0
    for i in range(len(power_set_o)):
        temp = pow(-1, len(power_set_o[i]))
        for j in range(1, candidate_num + 1):
            if j == c_i: continue
            temp = temp * formula_u_compute(power_set_o[i], set_o_neg, j)

        temp1 = 1
        power_set_union = power_set_o[i] + set_o_neg
        for k in range(len(power_set_union)):
            temp1 = temp1 * (1 - cond_prob_table.at[power_set_union[k], 'c' + str(c_i)])
        temp1 = temp1 - 1
        # print("temp1:",temp1)

        temp = temp * temp1
        sum_ = sum_ + temp

    return sum_

def option_cond_gradient_compute(set_o_pos, set_o_neg, o_i, c_i):
    o_i_str = 'o' + str(o_i)
    set_o_pos_remove = set_o_pos[:]
    if o_i_str in set_o_pos_remove:
        set_o_pos_remove.remove(o_i_str)

    # 生成幂集
    power_set_o_pos = generate_power_set(set_o_pos_remove)
    # print(power_set_o_pos)

    sum_ = 0
    for i in range(len(power_set_o_pos)):
        temp = pow(-1, len(power_set_o_pos[i]) + 1)

        temp_u = 1
        for j in range(1, candidate_num + 1):
            if j == c_i: continue
            new_set_part = power_set_o_pos[i] + [o_i_str]
            # print("new_set_part",new_set_part)
            temp_u = temp_u * formula_u_compute(new_set_part, set_o_neg, j)

        temp1 = 1
        new_set_part1 = power_set_o_pos[i] + set_o_neg
        if o_i_str in new_set_part1:
            new_set_part1.remove(o_i_str)

        for k in range(len(new_set_part1)):
            temp1 = temp1 * (1 - cond_prob_table.at[new_set_part1[k], 'c' + str(c_i)]) * candidate_prob[c_i - 1]

        # print("sum_:",sum_," temp:",temp," temp_u:",temp_u," temp1:",temp1)
        sum_ = sum_ + temp * temp_u * temp1

    return sum_

# 公式14 U_i的计算，与上面参数不同
def formula_u_compute_param(candidate_prob_, cond_prob_table_, set_o_part, set_o_neg, c_i):
    prob_ = 1
    set_o_union = set_o_part + set_o_neg
    # print("set_o_union",set_o_union)

    for i in range(len(set_o_union)):
        prob_ = prob_ * (1 - cond_prob_table_.at[set_o_union[i], 'c' + str(c_i)])

    # print("prob_ before:",prob_)
    # 传入的c_i是从1开始，而数组元素的下标是从0开始的
    prob_ = prob_ * candidate_prob_[c_i - 1] + 1 - candidate_prob_[c_i - 1]
    # print("prob_ after:",prob_)
    return prob_

# 公式13重写后的式子的计算，与上面参数不同
def formula_completion_param(candidate_prob_, cond_prob_table_, set_o_pos, set_o_neg):
    # 生成集合o的幂集
    power_set_o = generate_power_set(set_o_pos)

    sum_ = 0
    for i in range(len(power_set_o)):
        temp = pow(-1, len(power_set_o[i]))
        for j in range(candidate_num):
            temp = temp * formula_u_compute_param(candidate_prob_, cond_prob_table_, power_set_o[i], set_o_neg, j + 1)
        # print("temp:",temp)
        sum_ = sum_ + temp

    return sum_

def projected_gradient_method(param, l_bound, r_bound):
    if l_bound < param < r_bound:
        return param
    elif l_bound > param:
        return l_bound
    elif r_bound < param:
        return r_bound

def armijo_rule_eta(param_old, set_o_prob, diff, flag):
    beta = 0.1
    sigma = 0.01
    grad_ = diff / set_o_prob # 即梯度
    eta = 1
    param_new = param_old + eta * grad_

    set_o_positive = []
    set_o_negitive = []
    # ask all the feature
    for i in range(len(s_val)):
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

    l_bound = 0.0005
    r_bound = 0.99

    param_new = projected_gradient_method(param_old + eta * grad_, l_bound, r_bound)
    return param_new

def parameter_update_candidate(diff, set_o_prob, c_i, gamma_, set_o_pos, set_o_neg):
    sigma = 0.01
    beta = 0.9
    l_bound = 0.005
    r_bound = 0.980

    candidate_prob_temp = candidate_prob[:]
    cond_prob_table_temp = cond_prob_table[:]
    # 原参数值
    x = candidate_prob[c_i - 1]
    grad_ = diff / set_o_prob

    x_new = x + gamma_ * grad_
    x_old = x

    candidate_prob_temp[c_i - 1] = x_new
    gamma_new = gamma_
    iterations = 10

    if formula_completion_param(candidate_prob_temp, cond_prob_table_temp, set_o_pos,
                                set_o_neg) - set_o_prob >= sigma * grad_ * (x_new - x):
        while iterations:
            gamma_ = gamma_new
            gamma_new = gamma_new / beta
            x_new = projected_gradient_method(x + gamma_new * grad_, l_bound, r_bound)

            candidate_prob_temp[c_i - 1] = x_new
            print("x =", x, "x_new =", x_new)
            print("f(x_new) =",
                  formula_completion_param(candidate_prob_temp, cond_prob_table_temp, set_o_pos, set_o_neg), "f(x)",
                  set_o_prob)
            print("gamma_new =", gamma_new)
            if formula_completion_param(candidate_prob_temp, cond_prob_table_temp, set_o_pos, set_o_neg) - \
                    set_o_prob >= sigma * grad_ * (x_new - x) and x_new != x_old:
                # continue要把下面的内容提前做了
                x_old = x_new
                iterations -= 1
                continue
            else:
                break
    else:
        while iterations:
            gamma_new = gamma_new * beta
            x_new = x + gamma_new * grad_
            candidate_prob_temp[c_i - 1] = x_new
            if formula_completion_param(candidate_prob_temp, cond_prob_table_temp, set_o_pos, set_o_neg) - \
                    set_o_prob >= sigma * grad_ * (x_new - x) or x_new == x_old:

                gamma_ = gamma_new
                break
            else:
                iterations -= 1
                x_old = x_new
                continue

    param_new = projected_gradient_method(x + gamma_ * grad_, l_bound, r_bound)
    return param_new, gamma_


def parameter_update_option_cond_prob():
    return 1

def parameter_update(param_old, set_o_prob, diff, flag):
    grad_ = diff / set_o_prob

    gamma = 0.01
    l_bound = 0.0005
    r_bound = 0.99

    if flag == 0:
        gamma = 0.01
    else:
        gamma = 0.02

    if flag == 0 or flag == 1:
        param_new = projected_gradient_method(param_old + gamma * grad_, l_bound, r_bound)
    else:
        param_new = projected_gradient_method(param_old - gamma * grad_, l_bound, r_bound)

    return param_new

gamma_cand = 1.0
gamma_op = 1.0

# 计算对一个志愿者提问以后，对于两个参数的更新
def each_worker_option_update(set_o_posi, set_o_neg, o_i, c_i):
    set_o_prob = formula_completion(set_o_posi, set_o_neg)

    diff1 = candiate_gradient_compute(set_o_posi, set_o_neg, c_i)
    diff2 = option_cond_gradient_compute(set_o_posi, set_o_neg, o_i, c_i)

    global gamma_cand
    res1 = parameter_update(candidate_prob[c_i - 1], set_o_prob, diff1, 0)
    # res1, gamma_cand = parameter_update_candidate(diff1, set_o_prob, c_i, gamma_cand, set_o_posi, set_o_neg)
    candidate_prob[c_i - 1] = res1
    # print(candidate_prob)
    # print("gamma_cand =",gamma_cand)

    # 分别来更新
    if len(set_o_posi) == 0:
        res2 = 1 - parameter_update(1 - cond_prob_table.at['o' + str(o_i), 'c' + str(c_i)], set_o_prob, diff2, 2)
    else:
        res2 = 1 - parameter_update(1 - cond_prob_table.at['o' + str(o_i), 'c' + str(c_i)], set_o_prob, diff2, 1)

    cond_prob_table.at['o' + str(o_i), 'c' + str(c_i)] = res2
    # print(cond_prob_table)

test_set_pos = ['o1']
test_set_neg = []

# 测试代码
def test():
    print("Test Start...")
    for i in range(9):
        # print("test_set_pos",test_set_pos)
        set_o_prob = formula_completion(test_set_pos, test_set_neg)
        # print("set_o_prob",set_o_prob)
        diff1 = candiate_gradient_compute(test_set_pos, test_set_neg, 2)
        # print("diff1:",diff1)
        diff2 = option_cond_gradient_compute(test_set_pos, test_set_neg, 3, 2)
        # print("diff2:",diff2)

        res1 = parameter_update(candidate_prob[1], set_o_prob, diff1)
        candidate_prob[1] = res1
        print(candidate_prob)

        res2 = 1 - parameter_update(1 - cond_prob_table.at['o3', 'c2'], set_o_prob, diff2)
        cond_prob_table.at['o3', 'c2'] = res2
        print(cond_prob_table)
        print()
    print("Test End...")


# 数据集
fileHeader = '../Synthetic_data/'

data_Filename = fileHeader + 'sys_data1.txt'
history_Filename = fileHeader + 'sys_data_history1.txt'

rou_arr = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
rou = 0

cand_tuple = set()
op_tuple = set()

# 创建表格，其中元素均初始化为0
cand_list = []
op_list = []
option_dict = {}

# 处理数据、图结构
def process_graph1():
    with open(data_Filename, 'r') as file:
        for line in file:
            numbers_str = line.split()
            numbers = [int(num) for num in numbers_str]
            # print(numbers)

            cand_tuple.add(numbers[0])
            op_tuple.add(numbers[1])
    # print(cand_tuple,op_tuple)

    for i in range(1, len(cand_tuple) + 1):
        cand_list.append('c' + str(i))
    for j in range(1, len(op_tuple) + 1):
        op_list.append('o' + str(j))
        option_dict['o' + str(j)] = set()
    # print(cand_list,op_list)
    # print(option_dict)

process_graph1()

# 指定列名和行名
columns = cand_list
index = op_list

zero_data = np.zeros((len(index), len(columns)))
cond_prob_table = pd.DataFrame(zero_data, index=index, columns=columns)

# 每个candidate的概率为1/n，存储的是每个candidate为真的概率
candidate_prob = np.full(len(columns), 1 / len(columns))
# print(candidate_prob)
candidate_num = len(candidate_prob)

# 读入历史观测数据
history_data = []
history_data_num = 0

# 处理数据、图结构
def process_graph2():
    with open(data_Filename, 'r') as file:
        for line in file:
            numbers_str = line.split()
            numbers = [int(num) for num in numbers_str]
            # 更新选项的字典，记录选项与哪些候选之间存在边
            option_dict['o' + str(numbers[1])].add('c' + str(numbers[0]))

            cond_prob_table.at['o' + str(numbers[1]), 'c' + str(numbers[0])] = 0.5
    # print(cond_prob_table)
    # print("option_dict:",option_dict)

    with open(history_Filename, 'r') as file:
        for line in file:
            numbers_str = line.split()
            numbers = [int(num) for num in numbers_str]
            history_data_num = len(numbers)
            history_data.append(numbers)
    # 读入的history_data数组是一个多维数组
    # print(history_data)
    # print(history_data_num)

process_graph2()

# 对于迭代次数的对比，分别记录迭代次数为5、10、15、20次情况下的结果
def write_to_file(file_path, content):
    # 确保目录存在
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 以追加模式打开文件，如果文件不存在则创建文件
    with open(file_path, 'a') as file:
        file.write(content + '\n')  # 添加内容并换行

# 按照自然顺序为提问顺序，对于每个问题，从其对应的历史观测数据中随机采样一个 作为当前志愿者的回答
# iteration是迭代次数;
def sample_random(iteration):
    # print("Sample random...")

    try:
        # 迭代iteration次
        for s in range(iteration):
            # print("Iteration -",s+1)

            for t in range(len(op_list)):
                worker_answer = random.choice(history_data[t])
                print("s-", s, " answer-", worker_answer)

                set_o_positive = []
                set_o_negitive = []
                if worker_answer == 1:
                    set_o_positive.append('o' + str(t + 1))
                else:
                    set_o_negitive.append('o' + str(t + 1))

                for key in option_dict:
                    if key == 'o' + str(t + 1):
                        for element in option_dict[key]:
                            cand_seq_ = int(element[1:])
                            each_worker_option_update(set_o_positive, set_o_negitive, t + 1, cand_seq_)

                            for u in range(len(candidate_prob)):
                                if abs(candidate_prob[u] - 1) <= 0.02:
                                    # print("Exist a candidate closed to 1...")
                                    raise BreakLoop
                                elif candidate_prob[u] <= 0:
                                    # print("Exist a candidate less than 0...")
                                    raise BreakLoop
                        break
                print("s-", s)
                print()
                print("candidate prob: ", candidate_prob)
                print("conditional probabilistic： ")
                print(cond_prob_table)
                print()

        # print()

    except BreakLoop:
        print("Exited all loops")
        print("Iteration", s + 1, "times ending...")


# 特征价值矩阵，并对特征价值矩阵进行初始化
s_val = [1 / len(op_list) for _ in range(len(op_list))]

def search_maximal_element(array_, flag_):
    # array_ is the raw array, flag_ is used to mark whether the element is used
    array_index = np.zeros(len(array_))
    for i in range(len(array_)):
        array_index[i] = i + 1

    for i in range(len(array_)):
        for j in range(len(array_)):
            if array_[i] < array_[j]:
                t = array_[j]
                t1 = flag_[j]
                t2 = array_index[j]
                array_[j] = array_[i]
                flag_[j] = flag_[i]
                array_index[j] = array_index[i]
                array_[i] = t
                flag_[i] = t1
                array_index[i] = t2

    for i in range(len(array_)):
        if flag_[i] == 1:
            # 返回满足要求的特征编号
            return int(array_index[i])
    # 异常情况
    return -1

# 创建一个指定行数和列数的二维列表。
def create_2d_list(rows, cols, initial_value=0):
    return [[initial_value for _ in range(cols)] for _ in range(rows)]

def update_value_preference_matrix(matrix_, new_observation):
    for i in range(len(new_observation)):
        for j in range(len(new_observation)):
            if new_observation[i] == 1 and new_observation[j] == 0:
                matrix_[i][j] = 1
            else:
                matrix_[i][j] = 0

    # 返回修改后的矩阵
    return matrix_

worker_matrix = []
it_matrix_ = create_2d_list(len(s_val), len(s_val))
worker_matrix.append(it_matrix_)
feature_pro = np.zeros(len(op_list))

# 得到回答，更新了候选边缘概率以及特征条件概率后，更新特征边缘概率
def update_feature_probability(feature_index):
    pro_temp = 0
    for j in range(len(candidate_prob)):
        pro_temp += cond_prob_table.at['o' + str(feature_index), 'c' + str(j + 1)] * candidate_prob[j]
    feature_pro[feature_index - 1] = pro_temp

# 计算公式中的c_phi
def compute_c_phi(matrix_):
    c_phi_ = 0
    for i in range(len(matrix_[0])):
        for j in range(len(matrix_[0])):
            if i != j:
                c_phi_ += matrix_[i][j]

    return c_phi_

# 计算公式中的D_ij
def compute_Dij(index1_, index2_):
    return (feature_pro[index1_] + feature_pro[index2_]) * (s_val[index1_] - s_val[index2_])

def compute_gradient_LS_si(s_index):
    formula_sum = 0

    for worker in range(len(worker_matrix)):
        formula_part1 = 0
        formula_part2 = 0
        formula_part3 = 0
        formula_part4 = 0
        formula_part5 = 0
        formula_part6 = 0

        # 首先将项分为多部分，先分别计算每一部分
        for j in range(len(op_list)):
            if j != s_index:
                formula_part1 += (feature_pro[s_index] + feature_pro[j]) * worker_matrix[worker][s_index][j]

        for i in range(len(op_list)):
            if i != s_index:
                formula_part2 += (feature_pro[i] + feature_pro[s_index]) * worker_matrix[worker][i][s_index]

        for i in range(len(op_list)):
            for j in range(len(op_list)):
                if i != j:
                    formula_part3 += math.exp(compute_Dij(i, j))

        for j in range(len(op_list)):
            if j != s_index:
                formula_part4 += (feature_pro[s_index] + feature_pro[j]) * math.exp(compute_Dij(s_index, j))

        for i in range(len(op_list)):
            if i != s_index:
                formula_part5 += (feature_pro[i] + feature_pro[s_index]) * math.exp(compute_Dij(i, s_index))

        formula_part6 = rou / 2 * s_val[s_index]
        # print(rou)

        formula_sum += formula_part1 - formula_part2 - \
                       (compute_c_phi(worker_matrix[worker]) / formula_part3 * (formula_part4 - formula_part5)) + formula_part6

    # 返回最后的梯度值
    return formula_sum

def formula_LSxita(s_m_):
    formula_sum = 0

    for worker in range(len(worker_matrix)):
        formula_part1 = 0
        formula_part2 = 0
        for i in range(len(op_list)):
            for j in range(len(op_list)):
                if i != j:
                    formula_part1 += worker_matrix[worker][i][j] * compute_Dij(i, j)

        for k in range(len(op_list)):
            for l in range(len(op_list)):
                if k != l:
                    formula_part2 += math.exp(compute_Dij(k, l))

        formula_sum += formula_part1 - compute_c_phi(worker_matrix[worker]) * math.log(formula_part2)

    # 返回计算出来的函数值
    return formula_sum

def armijo_rule_formula(s_m_, aerfa_):
    sigma = 0.01
    s_m1_ = s_m_ + aerfa_ * compute_gradient_LS_si(s_m_)
    return formula_LSxita(s_m1_) - formula_LSxita(s_m_) <= sigma * compute_gradient_LS_si(s_m_) * (s_m1_ - s_m_)

def matrix_fusion(matrix_):
    for i in range(len(matrix_[0])):
        for j in range(len(matrix_[0])):
            worker_matrix[0][i][j] += matrix_[i][j]

# 根据梯度值更新sm
def update_si():
    eta = 0.01
    lower_bound = 0.0005
    high_bound = 0.9995
    for i in range(len(op_list)):
        new_param_ = s_val[i] + eta * compute_gradient_LS_si(i)
        s_val[i] = projected_gradient_method(new_param_, lower_bound, high_bound)

def biLevel_strategy(iterations, vote_num, likelihood_result_avg, hsr_result_avg):
    # 记录结果的中间值
    likelihood_result_tmp = []
    hsr_result_tmp = []

    try:
        iterations_expected = 0 # 设置预期的停止迭代次数

        obs_result_iter_pos = []
        obs_result_iter_neg = []

        for it in range(iterations):
            # 标记数组中，如果元素为1，表示该元素是可用的，即没有被提问过
            s_val_flag = np.ones(len(s_val), dtype=int)
            it_observation = np.zeros(len(s_val), dtype=int)
            it_matrix = create_2d_list(len(s_val), len(s_val))
            worker_matrix.append(it_matrix)

            # ask all the feature
            for i in range(len(s_val)):
                # print(s_val)
                s_val_temp = copy.deepcopy(s_val)
                s_val_flag_temp = copy.deepcopy(s_val_flag)
                f_index = search_maximal_element(s_val_temp, s_val_flag_temp)
                s_val_flag[f_index - 1] = 0

                worker_answer = random.choice(history_data[f_index - 1])
                vote_num += 1
                it_observation[f_index - 1] = worker_answer
                # 更新价值偏好矩阵
                update_value_preference_matrix(it_matrix, it_observation)
                worker_matrix[len(worker_matrix) - 1] = it_matrix

                set_o_positive = []
                set_o_negitive = []

                if worker_answer == 1:
                    set_o_positive.append('o' + str(f_index))
                    # 记录所有的结果
                    obs_str = 'o' + str(f_index)
                    if it==0:
                        obs_result_iter_pos.append(obs_str)
                    else:
                        if obs_str in obs_result_iter_neg:
                            obs_result_iter_neg.remove(obs_str)
                            obs_result_iter_pos.append(obs_str)
                            # 否则这个元素就应该已经在正集合中了
                else:
                    set_o_negitive.append('o' + str(f_index))
                    obs_str = 'o' + str(f_index)
                    if it==0:
                        obs_result_iter_neg.append(obs_str)
                    else:
                        if obs_str in obs_result_iter_pos:
                            obs_result_iter_pos.remove(obs_str)
                            obs_result_iter_neg.append(obs_str)

                """
                if worker_answer == 1:
                    set_o_positive.append('o' + str(f_index))
                    # 记录所有的结果
                    obs_str = 'o' + str(f_index)
                    if it==0 or it==1:
                        obs_result_iter_pos.append(obs_str)
                    else:
                        if obs_str in obs_result_iter_neg:
                            obs_result_iter_neg.remove(obs_str)
                            obs_result_iter_pos.append(obs_str)
                        else:
                            if obs_result_iter_pos.count(obs_str) < 2:
                                obs_result_iter_pos.append(obs_str)
                else:
                    set_o_negitive.append('o' + str(f_index))
                    obs_str = 'o' + str(f_index)
                    if it==0 or it==1:
                        obs_result_iter_neg.append(obs_str)
                    else:
                        if obs_str in obs_result_iter_pos:
                            obs_result_iter_pos.remove(obs_str)
                            obs_result_iter_neg.append(obs_str)
                        else:
                            if obs_result_iter_neg.count(obs_str) < 2:
                                obs_result_iter_neg.append(obs_str)
                """

                obs_result_tmp = 1 - formula_completion(obs_result_iter_pos, obs_result_iter_neg)

                for key in option_dict:
                    if key == 'o' + str(f_index):
                        for element in option_dict[key]:
                            cand_seq_ = int(element[1:])

                            each_worker_option_update(set_o_positive, set_o_negitive, f_index, cand_seq_)

                            if iterations_expected == 0:
                                for u in range(len(candidate_prob)):
                                    if abs(candidate_prob[u] - 1) <= 0.02:
                                        # print("Exist a candidate closed to 1...")
                                        # raise BreakLoop
                                        # iterations_expected = 5 * it
                                        iterations_expected = 200
                                        print(iterations_expected," is get.")
                                        break
                                    elif candidate_prob[u] <= 0:
                                        print("Exist a candidate less than 0...")
                                        raise BreakLoop
                            elif(it != 0 and it == iterations_expected):
                                print("Already achieved expected iterations.")
                                raise BreakLoop  # 当满足条件时，抛出自定义异常退出所有循环
                        break
                # print()

                update_value_preference_matrix(it_matrix, it_observation)
                update_feature_probability(f_index)
                update_si()

                # 写入集合中
                likelihood_result_tmp.append(1-formula_completion(obs_result_iter_pos,obs_result_iter_neg))
                # write_to_file("../exp_bilevel/exp_conv/data1_rou005_likelihood.txt",str( 1-formula_completion(obs_result_iter_pos,obs_result_iter_neg) ))

                # print(iterations_expected,max(candidate_prob))
                # write_to_file("../exp_bilevel/exp_conv/data1_rou005_hsr.txt",str(max(candidate_prob)))
                hsr_result_tmp.append(max(candidate_prob))

                # if (i == 0):
                #     print("Exit test....")
                #     raise BreakLoop

            matrix_fusion(it_matrix)
            worker_matrix.pop()

            # print(len(likelihood_result_tmp),len(hsr_result_tmp))
            # print("the end: ", len(worker_matrix))

    except BreakLoop:
        likelihood_result_avg.append(likelihood_result_tmp)
        hsr_result_avg.append(hsr_result_tmp)

        print("Exited all loops")
        print("Iteration", it + 1, "times ending...")
        return vote_num, likelihood_result_avg, hsr_result_avg


candidate_prob_backup = copy.deepcopy(candidate_prob[:])
cond_prob_table_backup = copy.deepcopy(cond_prob_table[:])


for iter_rou in range(len(rou_arr)):
    rou = rou_arr[iter_rou]

    worker_vote_num = 0
    vote_num_sum = 0
    runtime_use = 0

    likelihood_result_avg = []
    hsr_result_avg = []

    # 记录50次的结果
    for run_time in range(10):
        # 记录程序开始时间
        start_time = time.time()

        vote_num = 0
        worker_vote_num += 1

        worker_matrix.clear()
        worker_matrix.append(it_matrix_)

        for i in range(len(op_list)):
            pro_temp = 0
            for j in range(len(candidate_prob)):
                pro_temp += cond_prob_table.at['o' + str(i + 1), 'c' + str(j + 1)] * candidate_prob[j]
            feature_pro[i] = pro_temp

        s_val = [1 / len(op_list) for _ in range(len(op_list))]

        # 写入的文件目录
        # file_path = '../experiment_iter/sys_data3_res_iter20.txt'  # 替换为文件路径
        # file_path = '../experiment_real/galaxy_data20.txt'
        # print("can =",candidate_prob)

        vote_num, likelihood_result_avg, hsr_result_avg = biLevel_strategy(1000, vote_num, likelihood_result_avg, hsr_result_avg)
        print("vote_used_number: ", vote_num)
        vote_num_sum += vote_num

        # print(len(likelihood_result_avg),len(hsr_result_avg))

        # 展示更新后的结果
        # print("candidate prob: ", candidate_prob)
        # print("conditional probabilistic： ")
        # print(cond_prob_table)

        content = str(max(candidate_prob))  # 要写入的内容
        max_cand_index = candidate_prob.argmax() + 1
        print("probability_max candidate: ", max_cand_index)

        # write_to_file(file_path, content)
        runtime_temp = time.time() - start_time
        print("runtime: ", runtime_temp)
        runtime_use += runtime_temp

        # print("cand_bakcup =",candidate_prob_backup)
        candidate_prob = copy.deepcopy(candidate_prob_backup[:])
        cond_prob_table = copy.deepcopy(cond_prob_table_backup[:])

    for x in range(len(likelihood_result_avg[0])):
        calc_tmp1 = 0
        calc_tmp2 = 0
        for y in range(len(likelihood_result_avg)):
            calc_tmp1 += likelihood_result_avg[y][x]
            calc_tmp2 += hsr_result_avg[y][x]

        calc_tmp1 = calc_tmp1 / len(likelihood_result_avg)
        calc_tmp2 = calc_tmp2 / len(hsr_result_avg)

        content_str_likelihodd = "../exp_bilevel/exp_conv/data_real_rou" + str(iter_rou) + "_likelihood.txt"
        content_str_hsr = "../exp_bilevel/exp_conv/data_real_rou" + str(iter_rou) + "_hsr.txt"
        write_to_file(content_str_likelihodd, str(calc_tmp1))
        write_to_file(content_str_hsr, str(calc_tmp2))

# print("candidate prob: ",candidate_prob)
# print("conditional probabilistic： ")
# print(cond_prob_table)
# print("vote_average: ", vote_num_sum / worker_vote_num)
# print("runtime_average", runtime_use / worker_vote_num)


