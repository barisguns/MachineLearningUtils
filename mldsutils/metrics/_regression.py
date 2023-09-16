from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from mldsutils.model_selection import get_retained_y_test


def outlier_rmse_scorer(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    y_test = get_retained_y_test(estimator, y_test)
    score = (sqrt(mean_squared_error(y_test, y_pred)))
    return score


def outlier_r2_scorer(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    y_test = get_retained_y_test(estimator, y_test)
    score = r2_score(y_test, y_pred)
    return score

