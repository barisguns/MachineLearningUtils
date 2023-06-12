from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_array


class ShrinkageLDAClassifier(BaseEstimator, ClassifierMixin):
    """
    This is a wrapper class which wraps sklearn LDA Classifier with covariance matrix shrinkage (Ledoit-Wolf lemma).
    Covariance matrix shrinkage is an effective regularization technique for LDA especially when the
    input data has a large amount of variables (columns), relative to the number of data points (rows).
    """
    def __init__(self):
        """
        Construct a new 'LinearDiscriminantAnalysis' object with least squares
        (performed better compared to eigenvalue decomposition) as solver
        and covariance matrix shrinkage enabled.
        """
        self.estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, X, y):
        self.estimator.fit(X, y)

        # Return the classifier
        return self

    def predict(self, X):
        # Input validation
        X = check_array(X)
        y_pred = self.estimator.predict(X)
        return y_pred
