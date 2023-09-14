"""
This module contains outlier eliminators which inherits mldsutils.base.SamplerMixin.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import LocalOutlierFactor
from mldsutils.base import SamplerMixin
from copy import copy
from scipy.stats import f


class BasePlsOutlierElim(SamplerMixin, BaseEstimator):

    def __init__(self, conf_lev=0.95, pls_n_components=2, prediction=True):
        super().__init__()
        self.conf_lev = conf_lev
        self.prediction = prediction
        self.pls_n_components = pls_n_components
        self.pls = None
        self.test_x_scores = None
        self.test_x_loadings = None
        self.test_y_scores = None
        self.test_y_loadings = None

    def _fit_resample(self, X, y):
        pass

    def _resample(self, X, y=None):
        pass

    def window_outlier_check(self, data_point, window_matrix):
        X, indices_retained, outliers = self.resample(window_matrix)
        if data_point in X:
            return False
        else:
            return True

    def _get_test_scores_loadings(self, X):
        yt = self.pls.predict(X)
        self.pls.fit(X, yt)
        self.test_x_scores = self.pls.x_scores_
        self.test_x_loadings = self.pls.x_loadings_
        self.test_y_scores = self.pls.y_scores_
        self.test_y_loadings = self.pls.y_loadings_

    def _check_type_X_y(self, X, y):
        if not isinstance(X, np.ndarray):
            if isinstance(X, list):
                X = np.array(X)
            else:
                X = X.to_numpy()
        if y is not None:
            if not isinstance(y, np.ndarray):
                if isinstance(y, list):
                    y = np.array(y)
                else:
                    y = y.to_numpy()
        return X, y

    def _get_retained_indices_X_y(self, X, y, stat, stat_conf):
        indices_retained = []
        stat = stat.tolist()
        for i in range(len(stat)):
            if stat[i] < stat_conf:
                indices_retained.append(i)

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            y = y[indices_retained]

        X = X[indices_retained, :]

        return X, y, indices_retained


class QresPlsOutlierElim(BasePlsOutlierElim):

    def __init__(self, conf_lev=0.95, pls_n_components=2, prediction=True):
        super().__init__(conf_lev=conf_lev, pls_n_components=pls_n_components, prediction=prediction)
        return

    def _fit_resample(self, X, y):
        # Define PLS object
        self._check_type_X_y(X, y)

        self.pls = PLSRegression(n_components=copy(self.pls_n_components))
        # Fit data
        self.pls.fit(X, y)

        Q, Q_conf = self._get_Q_and_Qconf(X)

        X, y, self.indices_retained = self._get_retained_indices_X_y(X, y, Q, Q_conf)

        if self.prediction is True:
            self.pls.fit(X, y)
        return X, y
    
    def _resample(self, X, y=None):
        if self.prediction is True:
            X, y = self._check_type_X_y(X, y)

            self._get_test_scores_loadings(X)

            Q, Q_conf = self._get_Q_and_Qconf(X)

            X, y, self.indices_retained = self._get_retained_indices_X_y(X, y, Q, Q_conf)
            # Case of Prediction/Online Outlier Elimination
            
            if y is None:
                return (X,)
        return X, y

    def _get_Q_and_Qconf(self, X):
        # Get X scores
        T = self.pls.x_scores_
        # Get X loadings
        P = self.pls.x_loadings_
        # Calculate error array
        Err = X - np.dot(T, P.T)
        # Calculate Q-residuals (sum over the rows of the error array)
        Q = np.sum(Err ** 2, axis=1)

        # Estimate the confidence level for the Q-residuals
        i = np.max(Q) + 1
        while 1 - np.sum(Q > i) / np.sum(Q > 0) > self.conf_lev:
            i -= 1
        Q_conf = i

        return Q, Q_conf


class TsqPlsOutlierElim(BasePlsOutlierElim):

    def __init__(self, conf_lev=0.95, pls_n_components=2, prediction=True):
        super().__init__(conf_lev, pls_n_components, prediction)
        return

    def _fit_resample(self, X, y):
        # Define PLS object
        self._check_type_X_y(X, y)

        self.pls = PLSRegression(n_components=copy(self.pls_n_components))
        # Fit data
        self.pls.fit(X, y)

        Tsq, Tsq_conf = self._get_tsq_and_tsqconf(X, 0)

        X, y, self.indices_retained = self._get_retained_indices_X_y(X, y, Tsq, Tsq_conf)
        if self.prediction is True:
            self.pls.fit(X, y)
        return X, y

    def _resample(self, X, y=None):
        if self.prediction is True:
            self._check_type_X_y(X, y)

            self._get_test_scores_loadings(X)
            # Calculate Hotelling's T-squared (note that data are normalised by default)
            Tsq, Tsq_conf = self._get_tsq_and_tsqconf(X, 0.000001)

            X, y, self.indices_retained = self._get_retained_indices_X_y(X, y, Tsq, Tsq_conf)
            # Case of Prediction/Online Outlier Elimination
            if y is None:
                return (X,)
        return X, y

    def _get_tsq_and_tsqconf(self, X, eps):
        # Calculate Hotelling's T-squared (note that data are normalised by default)
        Tsq = np.sum((self.pls.x_scores_ / (np.std(self.pls.x_scores_, axis=0) + eps)) ** 2, axis=1)
        # Calculate confidence level for T-squared from the ppf of the F distribution
        Tsq_conf = f.ppf(q=self.conf_lev,
                         dfn=self.pls_n_components,
                         dfd=X.shape[0]) * self.pls_n_components * \
                        (X.shape[0] - 1) / (X.shape[0] - self.pls_n_components)
        return Tsq, Tsq_conf


class LOFResampler(LocalOutlierFactor, SamplerMixin, BaseEstimator):
    """
    Wrapper class that forms an mldsutils resampler out of sklearn.neighbors.LocalOutlierFactor.
    """
    def __init__(self,
                 n_neighbors=20,
                 *,
                 algorithm="auto",
                 leaf_size=30,
                 metric="minkowski",
                 p=2,
                 metric_params=None,
                 contamination="auto",
                 novelty=False,
                 n_jobs=None):
        super().__init__(n_neighbors=n_neighbors,
                         algorithm=algorithm,
                         leaf_size=leaf_size,
                         metric=metric,
                         p=p,
                         metric_params=metric_params,
                         contamination=contamination,
                         novelty=novelty,
                         n_jobs=n_jobs)

    def _fit_resample(self, X, y):
        self.fit(X, y)
        is_inlier = self.predict(X)
        self.indices_retained = np.argwhere(is_inlier == 1).reshape((1, -1))[0]
        X = X[is_inlier == 1]
        y = y[is_inlier == 1]
        return X, y

    def _resample(self, X, y=None):
        is_inlier = self.predict(X)
        self.indices_retained = np.argwhere(is_inlier == 1).reshape((1, -1))[0]
        X = X[is_inlier == 1]
        return (X,)
