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

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Generate synthetic regression dataset
X, y, coef = make_regression(n_samples=200, n_features=2, noise=20, coef=True, random_state=42)

ax.scatter(X[:, 0], X[:, 1], y, marker="o", c="blue")

# Add outliers to the target variable (y) to reduce the linear regression performance
outliers_x = np.random.normal(loc=0, scale=20, size=(10, 2))
outliers_y = np.random.normal(loc=0, scale=20, size=10)  # y values for outliers

ax.scatter(outliers_x[:, 0], outliers_x[:, 1], outliers_y, marker="o", c="orange")

X = np.concatenate([X, outliers_x])
y = np.concatenate([y, outliers_y])

n = 100

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# plt.show()


# pipelinezz = Pipeline([
#     # ("sc", StandardScaler()),
#     ("lof", LOFResampler(n_neighbors=20, novelty=True)),
#     # ("qres", QresPlsOutlierElim()),
#     ("lr", LinearRegression())
#     ]
# )
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, marker="o", c="blue")
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, marker="o", c="orange")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# pipelinezz.fit(X_train, y_train)
# inds_retained = pipelinezz.indices_retained
# ax.scatter(X_train[inds_retained, 0], X_train[inds_retained, 1], y_train, marker="o", c="blue")
# X_detected_outliers = np.delete(X_train, inds_retained)
# y_detected_outliers = np.delete(y_train, inds_retained)
# ax.scatter(X_detected_outliers[:, 0], X_detected_outliers[:, 1], y_detected_outliers, marker="o", c="orange")
# plt.show()

# y_pred = pipelinezz.predict(X_test)

# inds_retained = pipelinezz.indices_retained
# ax.scatter(X_test[inds_retained[0], 0], X_test[inds_retained[0], 1], y_test[inds_retained[0]], marker="o", c="blue")
# X_detected_outliers = np.delete(X_test, inds_retained[0], axis=0)
# y_detected_outliers = np.delete(y_test, inds_retained[0], axis=0)
# ax.scatter(X_detected_outliers[:, 0], X_detected_outliers[:, 1], y_detected_outliers, marker="o", c="orange")
# plt.show()

# print(y_pred)
# print(pipelinezz.indices_retained)


# cv_results = cross_val_score(pipelinezz, X, y, scoring=outlier_r2_scorer, cv=3)
# print(cv_results)


sc = StandardScaler()
lof = LOFResampler(n_neighbors=20, novelty=True)
lr = LinearRegression()

X_train_sc = sc.fit_transform(X_train)
X_train_lof, y_train_lof = lof.fit_resample(X_train_sc, y_train)
X_train_lof_cp = X_train_lof.copy()
y_train_lof_cp = y_train_lof.copy()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train_lof[:, 0], X_train_lof[:, 1], y_train_lof, marker="o", c="blue")
inds_retained = lof.indices_retained
X_detected_outliers = np.delete(X_train_lof_cp, inds_retained[0])
y_detected_outliers = np.delete(y_train_lof_cp, inds_retained[0])
ax.scatter(X_detected_outliers[:, 0], X_detected_outliers[:, 1], y_detected_outliers, marker="o", c="orange")
plt.show()


