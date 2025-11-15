# 实现将合成数据集中的每个数字+1，并且交换两个数字的位置

# 文件名
# file_path = '../Synthetic_data/sys_data15.txt'
file_path = "../Synthetic_data/sys_data5.txt"

# 读取数据
with open(file_path, 'r') as file:
    lines = file.readlines()

# 对每行的数据进行处理
modified_lines = []
for line in lines:
    # 拆分行中的数字
    numbers = line.split()

    # 交换两个数字的顺序
    temp = numbers[1]
    numbers[1] = numbers[0]
    numbers[0] = temp

    # 将两个数字都加1并转换回字符串
    new_numbers = [str(int(num) + 1) for num in numbers]

    # 将修改后的数字组合成一行，并添加到列表中
    modified_lines.append(' '.join(new_numbers) + '\n')

# 写回到同一个文件
with open(file_path, 'w') as file:
    file.writelines(modified_lines)

print("数据处理完成，已将结果写回同一个文件。")