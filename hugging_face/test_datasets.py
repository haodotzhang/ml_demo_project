# 1): git clone https://huggingface.co/datasets/lansinuote/ChnSentiCorp.git 手动下载好数据集，再执行下面代码

# 2) 下载真正的数据集
# #加载数据
# from datasets import load_dataset

# dataset = load_dataset(path='./ChnSentiCorp')
# print(dataset)

# #保存数据集到磁盘
# dataset.save_to_disk(dataset_dict_path='./data/ChnSentiCorp')

# 3) 直接从磁盘加载数据集
#从磁盘加载数据
from datasets import load_from_disk

dataset = load_from_disk('./data/ChnSentiCorp')
print(dataset)

#取出训练集
dataset = dataset['train']
print(dataset)
# print(dataset["features"], dataset["num_rows"]) #无法执行
#查看一个数据
a, b = dataset[0], dataset[1]
print(a, b) # 两条样本
print(len(dataset))
