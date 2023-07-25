from mldsutils.pipeline import Pipeline
from sklearn import pipeline as pp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression, make_blobs, make_moons
from sklearn.model_selection import cross_val_score
from mldsutils.preprocessing import LOFResampler
from mldsutils.metrics import outlier_rmse_scorer, outlier_r2_scorer
import numpy as np

# X, y = make_regression(n_samples=100, n_features=50, n_informative=40, random_state=42)

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
X = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0]

rng = np.random.RandomState(42)
X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
y = np.zeros(X.shape[0])

pipelinezz = Pipeline([
    ("sc", StandardScaler()),
    ("lof", LOFResampler()),
    ("lr", LinearRegression())
    ]
)

pipelinezz.fit(X, y)
y_pred = pipelinezz.predict(X)
print(y_pred)
print(pipelinezz.indices_retained)
