import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
# llama_values = [2.6, 3.89, 4.1, 1.9, 5.5, 6.15, 7.1, 3.6]
# fedprox_values = [5.3, 6.1, 4, 1.9, 5.2, 6.3, 6.7, 4.4]


llama_values = [1.4, 3, 2.6, 2, 2.56, 2.8, 3.5, 1.2]
fedprox_values = [1.2, 3.6, 2.6, 1.9, 3, 1.9, 2.8, 2.4]

print("llama_mean:{}".format(np.mean(llama_values)))
print("fedprox_mean:{}".format(np.mean(fedprox_values)))



# Number of variables
num_vars = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is a circle, so we need to "complete the loop" and append the start value to the end
llama_values += llama_values[:1]
fedprox_values += fedprox_values[:1]
angles += angles[:1]

# Draw the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, llama_values, color='blue', alpha=0.25, label='Llama2')
ax.fill(angles, fedprox_values, color='green', alpha=0.25, label='FedProx')

# Draw one line per variable and add labels
ax.plot(angles, llama_values, color='blue', linewidth=2, linestyle='solid')
ax.plot(angles, fedprox_values, color='green', linewidth=2, linestyle='solid')
ax.set_yticks([1, 3, 5])
ax.set_yticklabels(['1', '3', '5'])

# Add labels to the axes
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels,fontsize=16)

# Add title and legend
ax.set_title('Llama2 vs FedProx',fontsize='20')
ax.legend(loc='upper center',fontsize='large')

plt.show()
