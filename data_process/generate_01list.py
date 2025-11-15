# Initialization array size is 40, and all elements are 0.
array_size = 50
my_array = [0] * array_size

# Defines the location to be set to 1.
positions_to_set_one = [1,3,10,11,12]

# Set the element at a specific position to 1.
for pos in positions_to_set_one:
    my_array[pos - 1] = 1  # 减去1是因为列表索引从0开始

print(my_array)
