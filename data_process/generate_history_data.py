import random

# 产生n个0和1组成的序列，用空格分隔，其中1的个数的占比为x%
def generate_binary_sequence(n, x):
    # x 可以是百分比形式（如 20%）或小数形式（如 0.2）
    if isinstance(x, str) and '%' in x:
        x = float(x.strip('%')) / 100  # 将百分比转换为小数
    elif isinstance(x, (float, int)):
        x = float(x)

    # 计算 1 的数量
    num_ones = int(n * x)
    num_zeros = n - num_ones

    # 生成序列
    sequence = [1] * num_ones + [0] * num_zeros
    random.shuffle(sequence)  # 随机打乱序列

    # 将序列转换为字符串，并用空格分隔
    sequence_str = ' '.join(map(str, sequence))

    return sequence_str


# 测试generate_binary_sequence函数
def generate_binary_sequence_test():
    n = 20  # 序列长度

    x = '80%'  # 1 的占比
    for i in range(5):
        result = generate_binary_sequence(n, x)
        print(result)

    y = '20%'  # 1 的占比
    for i in range(7):
        result = generate_binary_sequence(n, y)
        print(result)

"""
该函数用于生成真实数据集的历史数据，其中
total_vote表示总的投票数
vote_list表示每个选项的投票数，这个中每个选项的顺序应该与GalaxyZoo.txt中组织的OC Model的顺序对应起来
history_num表示要产生的历史数据的数量
根据这些参数产生一个列表，表示每个选项需要生成的历史数据中1的比例
返回一个多维列表，其中每个列表表示一个选项的历史数据
"""
def generate_real_history_data(total_vote,vote_list,history_num):
    ratio = []
    for i in range(len(vote_list)):
        ratio.append(vote_list[i]/total_vote)
    # print(ratio)

    # 每个选项对应的比例有了之后，调用函数生成历史数据
    res = []
    for j in range(len(ratio)):
        temp = generate_binary_sequence(history_num, ratio[j])
        # 打开文件，将生成的内容写入文件
        with open("../Real_data/GalaxyZoo_history.txt", "a") as file:
            file.write(temp+"\n")
        res.append(temp)

    # 返回生成的历史数据
    return res

# total_vote = 44
# 分别表示o1、o2、o3、o4...的投票数
# vote_list = [2, 9, 9, 5, 5, 37, 33, 3, 3, 30, 30, 26, 26, 26, 30, 4]

# xx = generate_real_history_data(total_vote,vote_list,50)
# print(xx)

def generate_synthetic_history_data():
    # ans 4
    # ratio_list = [0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0]

    # ans 2
    # ratio_list = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # ans 8
    # ratio_list = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]

    # sys_data4
    # ratio_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    # 特征集合，如果该特征与正确答案有关，则为1
    ratio_list = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    # ratio_list = [1, 0, 0, 1, 1, 0, 0, 1, 0]

    for i in range(len(ratio_list)):
        if(ratio_list[i] == 0):
            sequence_str = generate_binary_sequence(20, 0.5)
        else:
            sequence_str = generate_binary_sequence(20, 0.5)

        print("sequence_str =",sequence_str)

        # with open("../experiment_diff_noise/data/GalaxyZoo_ans8_history.txt", "a") as file:
        # with open("../experiment_diff_noise/data/sys_data4_diff_noise.txt", "a") as file:
        # with open("../Synthetic_data/sys_data_multi_history150.txt", "a") as file:
        # with open("../Synthetic_data/sys_data_hist.txt", "a") as file:
        with open("sys_data_hist.txt", "a") as file:
        # with open("../Synthetic_data_mc_20/sys_data_history15.txt", "a") as file:
            file.write(sequence_str + "\n")

generate_synthetic_history_data()