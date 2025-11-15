# 定义文件名
filename = "../Real_data_mc/GalaxyZoo.txt"

# 读取文件内容
with open(filename, "r") as file:
    lines = file.readlines()  # 读取所有行

# 处理每一行
modified_lines = []
for line in lines:
    # 去除首尾空格并拆分每行的数字
    numbers = line.strip().split(" ")

    # 将字符串转为整数，并对每个数字减1
    num1 = int(numbers[0]) - 1
    num2 = int(numbers[1]) - 1

    # 交换两部分的内容
    new_line = f"{num2} {num1}\n"

    # 添加到修改后的列表
    modified_lines.append(new_line)

# 输出交换后的内容
# print(modified_lines)

# 将修改后的内容写回到同一个文件
with open(filename, "w") as file:
    file.writelines(modified_lines)  # 写入修改后的行