"""
Implements SamplerMixin class which extends imblearn.base.SamplerMixin.
It currently overrides fit_resample() method, in order to disable input checks, this will be removed soon.
It implements abstract class _resample() which should be overriden by custom resampler classes and should
contain the resampling logic.
Resamplers have an indices_retained attribute, which is intended to be set in resample() and/or fit_resample(),
in  order to expose the user the retained indices after resampling e.g. outlier elimination.
Cross validation and hyper-parameter optimization will only be possible if indices_retained are set by the resampler.

"""
from imblearn import base
from abc import abstractmethod

from .utils import ArraysTransformer

class SamplerMixin(base.SamplerMixin):

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy
        self.indices_retained = None

    def fit_resample(self, X, y):
        """Resample the dataset.

                Parameters
                ----------
                X : {array-like, dataframe, sparse matrix} of shape \
                        (n_samples, n_features)
                    Matrix containing the data which have to be sampled.

                y : array-like of shape (n_samples,)
                    Corresponding label for each sample in X.

                Returns
                -------
                X_resampled : {array-like, dataframe, sparse matrix} of shape \
                        (n_samples_new, n_features)
                    The array containing the resampled data.

                y_resampled : array-like of shape (n_samples_new,)
                    The corresponding label of `X_resampled`.
                """

        # TODO: input validation
        # TODO: parameter validation
        # Input validation and parameter validation of Imblearn is removed because they are specifically
        # tailored for classification tasks.
        arrays_transformer = ArraysTransformer(X, y)

        output = self._fit_resample(X, y)

        X_, y_ = arrays_transformer.transform(output[0], output[1])
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    @abstractmethod
    def _fit_resample(self, X, y):
        """Base method defined in each sampler to defined the sampling
        strategy.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        """
        pass

    def resample(self, X, y=None):
        """Resample the dataset in prediction. The indices of the retained data points will be stored in:
        self.indices_retained.

          Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : None
        For API consistency.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.
        """

        # TODO: input validation
        # TODO: parameter validation
        arrays_transformer = ArraysTransformer(X, y)
        output = self._resample(X, y)

        # TODO: indices_retained validation

        return arrays_transformer.transform(output[0], output[1]) if (
                len(output) == 2) else arrays_transformer.transform(output[0])

    @abstractmethod
    def _resample(self, X, y=None):
        """Base method defined in each sampler to do resampling in the prediction.
        If y is None, the indices of the retained data points must be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        """
        pass


