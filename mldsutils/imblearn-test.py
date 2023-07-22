from pipeline import Pipeline
from sklearn import pipeline as pp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from preprocessing import QresPlsOutlierElim, TsqPlsOutlierElim

X, y = make_regression(n_samples=100, n_features=50, n_informative=40, random_state=42)

pipelinezz = Pipeline([
    ("sc", StandardScaler()),
    ("qres", QresPlsOutlierElim()),
    ("tsq", TsqPlsOutlierElim()),
    ("lr", LinearRegression())
    ]
)

# pipe = pipeline.make_pipeline(StandardScaler(), LinearRegression())
# pipe = pp.make_pipeline(StandardScaler(), LinearRegression())

# cv_results = cross_val_score(pipe, X, y, scoring="r2", cv=3)

pipelinezz.fit(X, y)
y_pred = pipelinezz.predict(X)
print(y_pred)
print(pipelinezz.indices_retained)

# qres = QresPlsOutlierElim(conf_lev=0)
# qres.fit_resample(X, y)
# X = qres.resample(X)
# for i in X:
#     print(i)

# print(cv_results)
