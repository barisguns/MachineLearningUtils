"""
This module implements an example use of shrinkage LDA classifier from ml-utils.classification.
It imports Iris data set, scales it, trains a shrinkage LDA classifier on it and shows the accuracy.
It also plots the discriminators and data points.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA

from dsutils.classification import ShrinkageLDAClassifier

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply PCA to obtain the first two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Create an instance of Shrinkage LDA
s_lda = ShrinkageLDAClassifier()

# Fit the Shrinkage LDA model on the training data
s_lda.fit(X_train, y_train)

# Make predictions on the test data
y_pred = s_lda.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plotting the decision boundaries
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = s_lda.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)

# Plotting the data points
colors = ['navy', 'turquoise', 'darkorange']
target_names = data.target_names

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=0.8, color=color, label=target_name)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundaries with LSQR solver (PCA)')
plt.legend(loc='best')
plt.show()
