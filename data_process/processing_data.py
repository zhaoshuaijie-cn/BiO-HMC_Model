# 定义文件名
import copy

filename = '../Synthetic_data_mc_20/sys_data3_history'

ratio = [50,60,70,80,90]

for r in range(len(ratio)):
    filename_tmp = copy.deepcopy(filename)
    filename_tmp = filename_tmp + str(ratio[r]) + '.txt'
    print(filename_tmp)

    # 读取文件内容
    with open(filename_tmp, "r") as file:
        lines = file.readlines()  # 读取所有行

    # 复制每行数据两遍
    # modified_lines = [line.strip() + '\n' + line.strip() + '\n' + line.strip() + '\n' for line in lines]
    # 将','替换为', '
    modified_lines = [line.replace(" ", ", ") for line in lines]

    # 将修改后的内容写回到同一个文件
    with open(filename_tmp, "w") as file:
        file.writelines(modified_lines)  # 写入修改后的行