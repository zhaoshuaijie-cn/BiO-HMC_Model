# 初始化数组大小为40，全部元素为0
array_size = 50
my_array = [0] * array_size

# 定义要设置为1的位置
positions_to_set_one = [1,3,10,11,12]

# 将特定位置的元素设置为1
for pos in positions_to_set_one:
    my_array[pos - 1] = 1  # 减去1是因为列表索引从0开始

# 打印结果
print(my_array)
