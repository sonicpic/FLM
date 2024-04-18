from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
import sys
sys.path.append('/root/FLM')
sys.path.append("../../")  # 添加路径以便可以导入其他模块
from evaluate.test import test
from evaluate.test2 import test2

# 假设 labels_true 是根据数据集中的指令类型进行聚类得到的真实标签
# 假设 labels_pred 是根据模型权重聚类得到的预测标签

labels_true = test2()
labels_pred = test()


# 计算调整后的兰德指数
ari_score = adjusted_rand_score(labels_true, labels_pred)
print(f"Adjusted Rand Index: {ari_score}")

# 计算归一化互信息
nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
print(f"Normalized Mutual Information: {nmi_score}")

# 计算V-measure
v_measure = v_measure_score(labels_true, labels_pred)
print(f"V-measure: {v_measure}")
