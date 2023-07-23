import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from mldsutils.base import SamplerMixin
from copy import copy
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from scipy.stats import f


class QresPlsOutlierElim(SamplerMixin, BaseEstimator):

    def __init__(self, conf_lev=0.95, pls_n_components=2, prediction=True):
        super().__init__()
        self.conf_lev = conf_lev
        self.pls_n_components = pls_n_components
        self.pls = None
        self.Q_conf = None
        self.used = False
        self.prediction = prediction
        self.test_x_scores = None
        self.test_x_loadings = None
        self.test_y_scores = None
        self.test_y_loadings = None
        return

    def window_outlier_check(self, data_point, window_matrix):
        X, indices_retained, outliers = self.resample(window_matrix)
        if data_point in X:
            return False
        else:
            return True

    def _fit_resample(self, X, y):
        # Define PLS object
        if not isinstance(X, np.ndarray):
            if isinstance(X, list):
                X = np.array(X)
            else:
                X = X.to_numpy()
        if not isinstance(y, np.ndarray):
            if isinstance(y, list):
                y = np.array(y)
            else:
                y = y.to_numpy()

        self.pls = PLSRegression(n_components=copy(self.pls_n_components))
        # Fit data
        self.pls.fit(X, y)
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
        self.Q_conf = i

        indices_retained = []
        Q = Q.tolist()
        for i in range(len(Q)):
            if Q[i] < self.Q_conf:
                indices_retained.append(i)

        X = X[indices_retained, :]
        y = y[indices_retained]
        if self.prediction is True:
            self.pls.fit(X, y)
        return X, y
    
    def _resample(self, X, y=None):
        if self.prediction is True:
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

            yt = self.pls.predict(X)
            self.pls.fit(X, yt)
            self.test_x_scores = self.pls.x_scores_
            self.test_x_loadings = self.pls.x_loadings_
            self.test_y_scores = self.pls.y_scores_
            self.test_y_loadings = self.pls.y_loadings_
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
            self.Q_conf = i

            self.indices_retained = []
            outliers = []
            Q = Q.tolist()
            for i in range(len(Q)):
                if Q[i] < self.Q_conf:
                    self.indices_retained.append(i)
                else:
                    outliers.append(i)
            if not isinstance(X, np.ndarray):
                X = X.to_numpy()
            if y is not None:
                if not isinstance(y, np.ndarray):
                    y = np.array(y)
                y = y[self.indices_retained]

            X = X[self.indices_retained, :]
            # Case of Prediction/Online Outlier Elimination
            
            if y is None:
                return (X,)
        return X, y


class TsqPlsOutlierElim(SamplerMixin, BaseEstimator):

    def __init__(self, conf_lev=0.95, pls_n_components=2, prediction=True):
        super().__init__()
        self.conf_lev = conf_lev
        self.pls_n_components = pls_n_components
        self.pls = None
        self.Tsq = None
        self.Tsq_conf = None
        self.prediction = prediction
        self.test_x_scores = None
        self.test_x_loadings = None
        self.test_y_scores = None
        self.test_y_loadings = None
        return

    def window_outlier_check(self, data_point, window_matrix):
        X, indices_retained, outliers = self.resample(window_matrix)
        if data_point in X:
            return False
        else:
            return True

    def _fit_resample(self, X, y):
        # Define PLS object
        if not isinstance(X, np.ndarray):
            if isinstance(X, list):
                X = np.array(X)
            else:
                X = X.to_numpy()
        if not isinstance(y, np.ndarray):
            if isinstance(y, list):
                y = np.array(y)
            else:
                y = y.to_numpy()

        self.pls = PLSRegression(n_components=copy(self.pls_n_components))
        # Fit data
        self.pls.fit(X, y)
        # Calculate Hotelling's T-squared (note that data are normalised by default)
        self.Tsq = np.sum((self.pls.x_scores_ / np.std(self.pls.x_scores_, axis=0)) ** 2, axis=1)
        # Calculate confidence level for T-squared from the ppf of the F distribution
        self.Tsq_conf = f.ppf(q=self.conf_lev,
                              dfn=self.pls_n_components,
                              dfd=X.shape[0]) * self.pls_n_components * (X.shape[0] - 1) / (X.shape[0] - self.pls_n_components)

        indices_retained = []
        self.Tsq = self.Tsq.tolist()
        for i in range(len(self.Tsq)):
            if self.Tsq[i] < self.Tsq_conf:
                indices_retained.append(i)
        X = X[indices_retained, :]
        y = y[indices_retained]
        if self.prediction is True:
            self.pls.fit(X, y)
        return X, y

    def _resample(self, X, y=None):
        if self.prediction is True:
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

            yt = self.pls.predict(X)
            self.pls.fit(X, yt)
            self.test_x_scores = self.pls.x_scores_
            self.test_x_loadings = self.pls.x_loadings_
            self.test_y_scores = self.pls.y_scores_
            self.test_y_loadings = self.pls.y_loadings_
            # Calculate Hotelling's T-squared (note that data are normalised by default)
            self.Tsq = np.sum((self.pls.x_scores_ / (np.std(self.pls.x_scores_, axis=0) + 0.000001)) ** 2, axis=1)
            # Calculate confidence level for T-squared from the ppf of the F distribution
            self.Tsq_conf = f.ppf(q=self.conf_lev,
                                  dfn=self.pls_n_components,
                                  dfd=X.shape[0]) * self.pls_n_components * (X.shape[0] - 1) / (X.shape[0] - self.pls_n_components)

            self.indices_retained = []
            outliers = []
            self.Tsq = self.Tsq.tolist()
            for i in range(len(self.Tsq)):
                if self.Tsq[i] < self.Tsq_conf:
                    self.indices_retained.append(i)
                else:
                    outliers.append(i)
            if not isinstance(X, np.ndarray):
                X = X.to_numpy()
            if y is not None:
                if not isinstance(y, np.ndarray):
                    y = np.array(y)
                y = y[self.indices_retained]

            X = X[self.indices_retained, :]
            # Case of Prediction/Online Outlier Elimination
            if y is None:
                return (X,)
        return X, y
