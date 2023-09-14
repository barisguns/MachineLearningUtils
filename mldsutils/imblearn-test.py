from pipeline import Pipeline, make_pipeline
from sklearn import pipeline as pp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from preprocessing import QresPlsOutlierElim, TsqPlsOutlierElim, LOFResampler
from mldsutils.metrics import outlier_rmse_scorer, outlier_r2_scorer

X, y = make_regression(n_samples=100, n_features=50, n_informative=40, random_state=42)

pipelinezz = Pipeline([
    ("sc", StandardScaler()),
    ("qres", QresPlsOutlierElim()),
    ("tsq", TsqPlsOutlierElim()),
    ("lr", LinearRegression())
    ]
)

pipe = make_pipeline(StandardScaler(), LOFResampler(n_neighbors=2, novelty=True), LinearRegression())
# pipe = pp.make_pipeline(StandardScaler(), LinearRegression())

# cv_results = cross_val_score(pipelinezz, X, y, scoring=outlier_r2_scorer, cv=3)
# print(cv_results)

pipe.fit(X, y)
y_pred = pipe.predict(X)
print(y_pred)
print(pipe.indices_retained)

# qres = QresPlsOutlierElim(conf_lev=0)
# qres.fit_resample(X, y)
# X = qres.resample(X)
# for i in X:
#     print(i)

# print(cv_results)
