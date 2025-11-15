import random

# Generate n sequences of 0s and 1s, separated by spaces, where the number of 1s accounts for x%.
def generate_binary_sequence(n, x):
    # X can be a percentage (such as 20%) or a decimal (such as 0.2).
    if isinstance(x, str) and '%' in x:
        x = float(x.strip('%')) / 100  # Convert percentages to decimals
    elif isinstance(x, (float, int)):
        x = float(x)

    # Calculate the number of 1.
    num_ones = int(n * x)
    num_zeros = n - num_ones

    # Generating sequence
    sequence = [1] * num_ones + [0] * num_zeros
    random.shuffle(sequence)  # 随机打乱序列

    # Converts a sequence into a string separated by spaces.
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
This function is used to generate historical data of real data sets, where
Total_vote represents the total number of votes.
Vote_list indicates the number of votes for each option, and the order of each option in this list should correspond to the order of OC Model organized in GalaxyZoo.txt.
History_num indicates the amount of historical data to be generated.
According to these parameters, a list is generated, indicating the proportion of 1 in the historical data that each option needs to generate.
Returns a multidimensional list, where each list represents the historical data of an option.
"""
def generate_real_history_data(total_vote,vote_list,history_num):
    ratio = []
    for i in range(len(vote_list)):
        ratio.append(vote_list[i]/total_vote)
    # print(ratio)

    # After the proportion corresponding to each option is known, the function is called to generate historical data.
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