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
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys

sys.path.append('/root/FLM')
sys.path.append("../../")  # 添加路径以便可以导入其他模块


def test():
    parent_folder = '../lora-shepherd/100/0'  # 修改为你的文件夹路径

    # # 设置你想访问的层的键名
    # layer_key = 'base_model.model.model.layers.31.self_attn.q_proj.lora_A.weight'

    # # 想要获取权重的层的键名列表
    # layer_keys = [
    #     'base_model.model.model.layers.28.self_attn.q_proj.lora_A.weight',
    #     'base_model.model.model.layers.28.self_attn.q_proj.lora_B.weight',
    #     'base_model.model.model.layers.29.self_attn.q_proj.lora_A.weight',
    #     'base_model.model.model.layers.29.self_attn.q_proj.lora_B.weight',
    #     'base_model.model.model.layers.30.self_attn.q_proj.lora_A.weight',
    #     'base_model.model.model.layers.30.self_attn.q_proj.lora_B.weight',
    #     'base_model.model.model.layers.31.self_attn.q_proj.lora_A.weight',
    #     'base_model.model.model.layers.31.self_attn.q_proj.lora_B.weight',
    # ]

    # 动态生成键名列表
    layer_keys = []
    for i in range(32):  # 从0到31，共32层
        layer_keys.append(f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight')
        layer_keys.append(f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight')

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

            # 用于存储当前模型的所有层权重
            all_flattened_weights = []

            # 检查并处理每个层的权重
            for layer_key in layer_keys:
                if layer_key in model_weights:
                    # 获取特定层的权重并从GPU转移到CPU
                    specific_layer_weights = model_weights[layer_key].cpu()
                    # print(specific_layer_weights.shape)
                    # 展平权重
                    flattened_weights = specific_layer_weights.flatten().numpy()
                    # print(flattened_weights.shape)
                    all_flattened_weights.append(flattened_weights)
                else:
                    print(f"Layer {layer_key} not found in model {i}.")

            # 将所有层的权重拼接成一个长向量
            concatenated_weights = np.concatenate(all_flattened_weights)
            vectors.append(concatenated_weights)

        else:
            print(f"Model file not found for model {i}.")

    # 检查是否所有模型都被正确读取和转换
    if len(vectors) == 100:
        # 进行k-means聚类
        kmeans = KMeans(n_clusters=5, random_state=0).fit(vectors)
        labels = kmeans.labels_

        # 进行谱聚类
        # spectral = SpectralClustering(n_clusters=10, random_state=0, affinity='nearest_neighbors')
        # labels = spectral.fit_predict(vectors)

        # # 进行DBSCAN聚类
        # dbscan = DBSCAN(eps=0.5, min_samples=5)  # 这里的eps和min_samples根据数据调整
        # labels = dbscan.fit_predict(vectors)

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

    return labels


if __name__ == '__main__':
    test()





