from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the breast cancer dataset
cancer = load_breast_cancer()
scaler = StandardScaler()

# Standardize the data
df_scaled = scaler.fit_transform(cancer.data)

# Convert the data to a DataFrame for better handling in visualization
df_scaled = pd.DataFrame(data=df_scaled, columns=cancer.feature_names)

# Plot histograms for all features to visualize distributions
df_scaled.hist(bins=20, figsize=(10, 10))
plt.show()

# Apply PCA with 3 components
pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_scaled)

# Print the shape of the transformed data
print("Shape of df_pca:", df_pca.shape)

# Explained variance ratio from PCA
ex_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio:", ex_variance_ratio)

# Scatter plots of the first vs. third and second vs. third principal components side by side
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

# First and second principal components
Xax = df_pca[:, 0]
Yax = df_pca[:, 1]
labels = cancer.target
cdict = {0: 'red', 1: 'green'}
labl = {0: 'Malignant', 1: 'Benign'}
marker = {0: '*', 1: 'o'}
alpha = {0: 0.5, 1: 0.7}

for l in np.unique(labels):
    ix = np.where(labels == l)
    ax[0].scatter(Xax[ix], Yax[ix], c=cdict[l], s=40, label=labl[l], marker=marker[l], alpha=alpha[l])
ax[0].set_xlabel("First Principal Component", fontsize=14)
ax[0].set_ylabel("Second Principal Component", fontsize=14)
ax[0].legend()

# First and third principal components
Yax = df_pca[:, 2]
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
s = sns.heatmap(cancer_df[feature_worst].corr(), cmap='coolwarm', annot=True)
s.set_yticklabels(s.get_yticklabels(), rotation=30, fontsize=7)
s.set_xticklabels(s.get_xticklabels(), rotation=30, fontsize=7)
plt.show()
