import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
customer_data = pd.read_csv('shopping_data.csv')

# Keep only 'Annual Income' and 'Spending Score' columns
data = customer_data.iloc[:, 3:5].values

# Create linkage matrix using 'Average' linkage method
linked = linkage(data, method='average')

# Initialize a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))  # 1 row, 2 columns for side-by-side display

# TASK 1: Dendrogram
ax1.set_title('Dendrogram for Shopping Data')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, ax=ax1)
ax1.set_xlabel('Customers')
ax1.set_ylabel('Euclidean Distance')

# TASK 2: Perform Agglomerative Clustering
n_clusters = 5  
cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')  
y_pred = cluster.fit_predict(data)

# Scatter plot for clustering
ax2.scatter(data[:, 0], data[:, 1], c=y_pred, cmap='rainbow', s=100)
ax2.set_title(f'Clusters of customers (n_clusters={n_clusters})')
ax2.set_xlabel('Annual Income (k$)')
ax2.set_ylabel('Spending Score (1-100)')
ax2.grid(True)  

plt.tight_layout()  
plt.show()


# TASK 3: Conclusion
# By analyzing the plot, we can draw conclusions about customer behavior:
# 1. High-income, high-spending customers.
# 2. Low-income, moderate-spending customers.
# 3. Moderate-income customers with varying spending habits.
# These insights can be used for targeted marketing strategies, promotions, or customer engagement.
