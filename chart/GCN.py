import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
from tqdm import tqdm  # 用于进度条展示
import json

def create_graph_data(vectors):
    """ 创建一个简单的全连接图。"""
    num_nodes = len(vectors)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.conv3 = GCNConv(output_dim, input_dim)  # 添加一个新层以匹配输入维度

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)  # 输出与输入同样的维度
        return x


def test():
    parent_folder = '../lora-shepherd/100/0'
    layer_keys = []
    for i in range(30,32):
        layer_keys.append(f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight')
        layer_keys.append(f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight')

    vectors = []
    for i in tqdm(range(100)):
        folder_name = f"local_output_{i}"
        folder_path = os.path.join(parent_folder, folder_name)
        model_file = os.path.join(folder_path, 'pytorch_model.bin')
        if os.path.exists(model_file):
            model_weights = torch.load(model_file)
            all_flattened_weights = [model_weights[layer_key].cpu().flatten().numpy() for layer_key in layer_keys if layer_key in model_weights]
            concatenated_weights = np.concatenate(all_flattened_weights)
            vectors.append(concatenated_weights)

    vectors = np.array(vectors)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(vectors, dtype=torch.float).to(device)
    edge_index = create_graph_data(vectors).to(device)

    model = GCN(input_dim=vectors.shape[1], hidden_dim=64, output_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data = Data(x=x, edge_index=edge_index)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)  # 假设的损失函数
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()

    spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors')
    labels = spectral.fit_predict(embeddings)
    print(labels)

    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(embeddings)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    for i in range(len(reduced_vectors)):
        ax.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], color=colors[labels[i]])
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.title('3D PCA Projection of GNN Embeddings')
    plt.savefig('gnn_cluster_plot.png')
    plt.close()

    return labels

if __name__ == '__main__':
    test()
