from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the breast cancer dataset
cancer = load_breast_cancer()

# Scaling the data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(cancer.data)

# Apply PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Print the shape of the transformed data
print("Shape of X_pca:", X_pca.shape)

# Explained variance ratio
ex_variance = np.var(X_pca, axis=0)
ex_variance_ratio = ex_variance / np.sum(ex_variance)
print("Explained variance ratio:", ex_variance_ratio)

# Scatter plots of the first vs. third and second vs. third principal components side by side
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

# First and third principal components
Xax = X_pca[:, 0]
Yax = X_pca[:, 1]
labels = cancer.target
cdict = {0: 'red', 1: 'green'}
labl = {0: 'Malignant', 1: 'Benign'}
marker = {0: '*', 1: 'o'}
alpha = {0: 0.3, 1: 0.5}

for l in np.unique(labels):
    ix = np.where(labels == l)
    ax[0].scatter(Xax[ix], Yax[ix], c=cdict[l], s=40, label=labl[l], marker=marker[l], alpha=alpha[l])
ax[0].set_xlabel("First Principal Component", fontsize=14)
ax[0].set_ylabel("Second Principal Component", fontsize=14)
ax[0].legend()

# Second and third principal components
Xax = X_pca[:, 0]
Yax = X_pca[:, 2]
for l in np.unique(labels):
    ix = np.where(labels == l)
    ax[1].scatter(Xax[ix], Yax[ix], c=cdict[l], s=40, label=labl[l], marker=marker[l], alpha=alpha[l])
ax[1].set_xlabel("First Principal Component", fontsize=14)
ax[1].set_ylabel("Third Principal Component", fontsize=14)
ax[1].legend()

plt.tight_layout()
plt.show()

# Heatmap to visualize the contribution of features to each principal component
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1, 2], ['1st Comp', '2nd Comp', '3rd Comp'], fontsize=10)
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=65, ha='left')
plt.tight_layout()
plt.show()

# Correlation heatmap of the 'worst' features
feature_worst = list(cancer.feature_names[20:31])
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
s = sns.heatmap(cancer_df[feature_worst].corr(), cmap='coolwarm')
s.set_yticklabels(s.get_yticklabels(), rotation=30, fontsize=7)
s.set_xticklabels(s.get_xticklabels(), rotation=30, fontsize=7)
plt.show()