from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


def get_retained_y_test(estimator, y_test):
    if hasattr(estimator, "indices_retained") and \
            estimator.indices_retained is not None and len(estimator.indices_retained) != 0:
        if isinstance(estimator.indices_retained[0], list):
            for i in estimator.indices_retained:
                y_test = [y_test[x] for x in i]
        else:
            y_test = [y_test[x] for x in estimator.indices_retained]
    return y_test


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

