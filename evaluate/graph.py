import os
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def graph1():
    folder_path = '../data/100'  # 请替换为你的文件夹路径
    all_vectors = []
    all_categories = set()

    for i in range(100):
        file_name = f"local_training_{i}.json"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                counter = Counter(item['category'] for item in data if 'category' in item)
                all_categories.update(counter.keys())
                all_vectors.append(counter)
        else:
            print(f"File not found: {file_name}")
            all_vectors.append(Counter())  # Handle missing files by adding an empty counter

    category_list = sorted(all_categories)
    final_vectors = [[vector[cat] for cat in category_list] for vector in all_vectors]

    # Generate a stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 10))
    width = 0.35  # the width of the bars

    # We need to transform the data from final_vectors to be suitable for bar plotting
    data_transposed = np.array(final_vectors).T  # Transpose to get categories as first index

    bottom = np.zeros(100)  # Start stacking from zero
    for idx, category_data in enumerate(data_transposed):
        ax.bar(range(1, 101), category_data, width, bottom=bottom, label=category_list[idx])
        bottom += category_data  # Update the bottom to stack the next category on top of the previous one

    ax.set_xlabel('Client Index')
    ax.set_ylabel('Count')
    ax.set_title('Category distribution per client')
    ax.set_xticks(range(1, 101))
    ax.set_xticklabels(range(1, 101), rotation=90)
    ax.legend(title="Categories")

    plt.tight_layout()
    plt.savefig('category_distribution_per_client.png')
    plt.show()

if __name__ == '__main__':
    graph1()
