from mldsutils.pipeline import Pipeline
from sklearn import pipeline as pp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from mldsutils.preprocessing import LOFResampler
from mldsutils.preprocessing import QresPlsOutlierElim
from mldsutils.metrics import outlier_r2_scorer
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Generate synthetic regression dataset
X, y, coef = make_regression(n_samples=200, n_features=2, noise=20, coef=True, random_state=42)

# Add outliers to the target variable (y) to reduce the linear regression performance
outliers_x = np.random.normal(loc=0, scale=20, size=(10, 2))
outliers_y = np.random.normal(loc=0, scale=20, size=10)  # y values for outliers

X = np.concatenate([X, outliers_x])
y = np.concatenate([y, outliers_y])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

ax.scatter(X[:, 0], X[:, 1], y, marker="o")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


pipelinezz = Pipeline([
    # ("sc", StandardScaler()),
    ("lof", LOFResampler(n_neighbors=20, novelty=True)),
    # ("qres", QresPlsOutlierElim()),
    ("lr", LinearRegression())
    ]
)
#
# pipelinezz.fit(X_train, y_train)
# y_pred = pipelinezz.predict(X_test)
# print(y_pred)
# print(pipelinezz.indices_retained)


cv_results = cross_val_score(pipelinezz, X, y, scoring=outlier_r2_scorer, cv=3)
print(cv_results)

