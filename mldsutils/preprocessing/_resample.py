from sklearn.base import BaseEstimator
import numpy as np

from imblearn.base import SamplerMixin


class NumpyPytorchDatatypeResampler(SamplerMixin, BaseEstimator):
    """
    This is a custom imblearn resampler which basically transforms data
    into formats that would work as Pytorch inputs.
    It is especially useful when outputs of sklearn/imblearn transformers are to be given to Pytorch models as input.
    Imblearn is a library relying on sklearn that implements resampling tools and it is more useful here than sklearn
    because it can also transform "y" in the fit phase, which we need when preparing input for Pytorch.

    """
    def __init__(self):
        return

    def fit(self, X, y):
        return

    def fit_resample(self, X, y):
        return X.astype(np.float32), y.astype(np.int64)

    def transform(self, X, y=None):
        return X.astype(np.float32), None, None

    def _fit_resample(self, X, y):
        return X, y
