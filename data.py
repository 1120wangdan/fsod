import os
import random
import shutil

#自己写的创建数据集的文件  wd


source_folder1 = "/root/workspace/DeFRCN/datasets/coco/train2014"
source_folder2 = "/root/workspace/DeFRCN/datasets/coco/val2014"
destination_folder = "/root/workspace/DeFRCN/datasets/coco/trainval2014"

# # 创建目标文件夹
# os.makedirs(destination_folder, exist_ok=True)

# 获取源文件夹1中的所有文件
file_list1 = os.listdir(source_folder1)

# 获取源文件夹2中的所有文件
file_list2 = os.listdir(source_folder2)

# 打乱文件列表的顺序
random.shuffle(file_list1)
random.shuffle(file_list2)

# 遍历源文件夹1中的文件列表，将文件复制到目标文件夹中
for filename in file_list1:
    source_path = os.path.join(source_folder1, filename)
    destination_path = os.path.join(destination_folder, filename)
    shutil.copy2(source_path, destination_path)

# 遍历源文件夹2中的文件列表，将文件复制到目标文件夹中
for filename in file_list2:
    source_path = os.path.join(source_folder2, filename)
    destination_path = os.path.join(destination_folder, filename)
    shutil.copy2(source_path, destination_path)
print(1)


