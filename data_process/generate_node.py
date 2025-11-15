import random

# 设置左边和右边节点的范围
left_range = range(1, 81)  # 左边节点 1-20
right_range = range(1, 151)  # 右边节点 1-40

# 初始化一个字典来存储左边节点及其对应的边
bipartite_graph = {node: set() for node in left_range}

# 为每个左边节点随机生成6到8条边
for node in left_range:
    edges_count = random.randint(6, 8)  # 随机确定边的数量
    while len(bipartite_graph[node]) < edges_count:
        # 随机选取一个右边的节点
        right_node = random.choice(right_range)
        # 确保没有重复的边
        bipartite_graph[node].add(right_node)

# 验证所有特征是否都出现过
feature_list = []

# 设置文件名
file_path = '../Synthetic_data/sys_data_multi150.txt'

# 将结果写入到指定的txt文件
with open(file_path, 'w') as file:
    for left_node, right_nodes in bipartite_graph.items():
        for right_node in right_nodes:
            file.write(f"{left_node} {right_node}\n")
            feature_list.append(right_node)
            # print(left_node, right_node)

print(f"数据已成功生成并写入到 {file_path}。")
unique_list = list(set(feature_list))
print(unique_list)
print("特征长度为：",len(unique_list))