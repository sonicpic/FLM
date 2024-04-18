# 导入必要的库
import json
import argparse
import os
from tqdm import tqdm  # 用于进度条展示
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer  # 导入Hugging Face库中的模型和分词器
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from peft import set_peft_model_state_dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys

sys.path.append('/root/FLM')
sys.path.append("../../")  # 添加路径以便可以导入其他模块
from utils.conversation import get_conv_template  # 导入对话模板工具

parent_folder = '../lora-shepherd/100/0'  # 修改为你的文件夹路径

# 设置你想访问的层的键名
layer_key = 'base_model.model.model.layers.31.self_attn.q_proj.lora_A.weight'

vectors = []  # 用于存储展平后的权重

# 循环处理每个文件夹
for i in range(100):
    folder_name = f"local_output_{i}"
    folder_path = os.path.join(parent_folder, folder_name)
    model_file = os.path.join(folder_path, 'pytorch_model.bin')

    # 检查文件是否存在
    if os.path.exists(model_file):
        # 加载模型权重
        model_weights = torch.load(model_file)
        # print(f"Model {i}:")
        # print(model_weights)
        # 检查该键是否存在于权重中
        if layer_key in model_weights:
            # 获取特定层的权重
            # specific_layer_weights = model_weights[layer_key]
            # 获取特定层的权重并从GPU转移到CPU
            specific_layer_weights = model_weights[layer_key].cpu()
            print(f"Weights of {layer_key}:")
            # print(specific_layer_weights)
            print(specific_layer_weights.shape)
            print(specific_layer_weights.flatten())
            print(specific_layer_weights.flatten().shape)
            # print(type(specific_layer_weights))

            flattened_weights = specific_layer_weights.flatten().numpy()  # 展平并转换为numpy数组
            vectors.append(flattened_weights)
        else:
            print(f"Layer {layer_key} not found in the model.")
    else:
        print(f"Model file not found for model {i}.")

# 检查是否所有模型都被正确读取和转换
if len(vectors) == 100:
    # 进行k-means聚类
    kmeans = KMeans(n_clusters=10, random_state=0).fit(vectors)
    labels = kmeans.labels_
    print("Cluster labels for each vector:")
    print(labels)

else:
    print("Error: Some models were not processed correctly, expected 100 models but got:", len(vectors))

# 使用PCA进行降维到三维空间
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(vectors)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 为每个聚类分配不同的颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

for i in range(len(reduced_vectors)):
    ax.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], color=colors[labels[i]])

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('3D PCA Projection of the Vectors')
# 保存图像而不是显示
plt.savefig('cluster_plot.png')
plt.close()





