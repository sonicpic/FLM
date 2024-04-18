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

import os
import json
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans

def test2():
    # 文件夹路径
    folder_path = '../data/100'  # 替换为你的文件夹路径

    # 初始化存储每个文件类别计数的列表
    all_vectors = []

    # 初始化一个集合来存储所有见过的类别
    all_categories = set()

    # 遍历文件夹中的所有文件
    for i in range(100):
        file_name = f"local_training_{i}.json"
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否存在
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # 读取并解析JSON数据

                # 计算当前文件的类别统计
                counter = Counter(item['category'] for item in data if 'category' in item)

                # 将找到的类别添加到全局集合
                all_categories.update(counter.keys())

                # 存储当前文件的类别计数
                all_vectors.append(counter)
        else:
            print(f"File not found: {file_name}")
            all_vectors.append(Counter())  # 如果文件不存在，添加空的计数器

    # 转换计数器到向量
    category_list = sorted(all_categories)  # 对所有类别进行排序，确保顺序一致
    final_vectors = [[vector[cat] for cat in category_list] for vector in all_vectors]

    # 执行k-means聚类
    # kmeans = KMeans(n_clusters=5, random_state=0)
    # kmeans.fit(final_vectors)
    # labels = kmeans.labels_

    #谱聚类
    # spectral = SpectralClustering(n_clusters=10, random_state=0, affinity='nearest_neighbors')
    # labels = spectral.fit_predict(final_vectors)

    # 进行DBSCAN聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # 这里的eps和min_samples根据数据调整
    labels = dbscan.fit_predict(final_vectors)


    # 打印所有向量和对应的类别
    print("Categories found:", category_list)
    for index, vector in enumerate(final_vectors):
        print(f"Vector for client {index}: {vector}")

    # 打印聚类结果
    print("Cluster labels for each vector:")
    print(labels)

    # 使用PCA进行降维到三维空间
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(final_vectors)

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
    plt.savefig('cluster_plot2.png')
    plt.close()

    return labels

if __name__ == '__main__':
    test2()



