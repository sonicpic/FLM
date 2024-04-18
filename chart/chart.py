import json
from collections import Counter
import matplotlib.pyplot as plt

# Load the data from a JSON file
with open("../data/new-databricks-dolly-15k.json", "r", encoding="utf-8") as file:
    data_json = json.load(file)

# Extract categories and count them
categories = [item["category"] for item in data_json]
category_counts = Counter(categories)

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Category Distribution')

# Save the figure to the current directory
plt.savefig('category_distribution.png',dpi=300)
plt.close() # Close the figure to free up memory
