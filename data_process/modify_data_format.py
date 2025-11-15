"""
该文件的作用是读取文件，将每行历史数据中的分隔符由", "改为" "
"""

# 文件名
# file_path = '../Synthetic_data/sys_data_history15.txt'
# file_path = '../Synthetic_data/sys_data_hist.txt'
file_path = 'sys_data_hist.txt'

# 读取数据
with open(file_path, 'r') as file:
    lines = file.readlines()

# 替换每行中的", "为" "
modified_lines = [line.replace(', ', ' ') for line in lines]

# 写回到同一个文件
with open(file_path, 'w') as file:
    file.writelines(modified_lines)

print("数据处理完成，已将结果写回同一个文件。")
