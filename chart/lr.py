import numpy as np
import matplotlib.pyplot as plt

def compute_learning_rate(t, T, alpha_0, alpha_min, t_warmup):
    if t <= t_warmup:
        # 预热期内线性增加学习率
        return alpha_min + (alpha_0 - alpha_min) * t / t_warmup
    else:
        # 余弦退火调整学习率
        return alpha_min + 0.5 * (alpha_0 - alpha_min) * (1 + np.cos(np.pi * (t - t_warmup) / (T - t_warmup)))

# 设置参数
alpha_0 = 1.5e-4      # 初始学习率
alpha_min = 1.5e-5  # 最小学习率
T = 20            # 总轮次
t_warmup = 2      # 预热期轮次

# 计算每个轮次的学习率
learning_rates = [compute_learning_rate(t, T, alpha_0, alpha_min, t_warmup) for t in range(T)]

# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']  # Ensure Chinese characters display correctly
plt.figure(figsize=(10, 5))
plt.plot(learning_rates, label='学习率')
plt.xlabel('联邦学习通信轮次',fontsize=18)
plt.ylabel('学习率',fontsize=18)
plt.title('学习率变化曲线',fontsize=18)
plt.legend()
plt.grid(True)
plt.show()
