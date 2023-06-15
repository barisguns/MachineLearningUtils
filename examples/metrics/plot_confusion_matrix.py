"""
The aim of this script is to show the use of custom scorer confusion_matrix_scorer from mldsutils.metrics.
It uses the scorer in cross validation, and uses the cv result to plot the confusion matrix for each fold.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np

from mldsutils.metrics import confusion_matrix_scorer


# Load the Iris dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create a logistic regression classifier
clf = LogisticRegression()

# Perform cross-validation with the custom scorer
cv_results = cross_validate(clf, X, y, cv=5, scoring=confusion_matrix_scorer, return_estimator=True)

# Plot the confusion matrix for each fold
fig, axes = plt.subplots(1, len(cv_results['estimator']), figsize=(15, 4))

for ax, estimator, fold_idx in zip(axes, cv_results['estimator'], range(len(cv_results['estimator']))):
    cm = np.array([[cv_results[f'test_tn'][fold_idx], cv_results[f'test_fp'][fold_idx]],
                   [cv_results[f'test_fn'][fold_idx], cv_results[f'test_tp'][fold_idx]]])

    labels = data.target_names
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["P", "N"])
    ax.set_yticklabels(["P", "N"])

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    ax.set_title(f'Confusion Matrix (Fold #{fold_idx + 1})')

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

plt.tight_layout()
plt.show()
