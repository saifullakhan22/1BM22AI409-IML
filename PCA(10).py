#10

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply PCA to reduce the data to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the original data
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(X[y == i, 0], X[y == i, 1], color=colors[i], label=f'Iris-{i}')

plt.title('Original Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Plot the data after PCA
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=colors[i], label=f'Iris-{i}')

plt.title('Iris Dataset after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
