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

from sklearn.preprocessing import label_binarize
from sklearn.utils import parse_version
from sklearn.utils.multiclass import check_classification_targets

from imblearn.utils import check_sampling_strategy, check_target_type
from imblearn.utils._param_validation import validate_parameter_constraints
from imblearn.utils._validation import ArraysTransformer
import numpy as np


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
        # check_classification_targets(y)
        # arrays_transformer = ArraysTransformer(X, y)
        # X, y, binarize_y = self._check_X_y(X, y)

        # self.sampling_strategy_ = check_sampling_strategy(
        #     self.sampling_strategy, y, self._sampling_type
        # )

        X, y = self._fit_resample(X, y)

        # y_ = (
        #     label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        # )
        #
        # X_, y_ = arrays_transformer.transform(output[0], y_)
        # return (X_, y_) if len(output) == 2 else (X_, y_, output[2])
        return X, y

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
        # check_classification_targets(y)
        # arrays_transformer = ArraysTransformer(X, y)
        # X, y, binarize_y = self._check_X_y(X, y)
        #
        # self.sampling_strategy_ = check_sampling_strategy(
        #     self.sampling_strategy, y, self._sampling_type
        # )

        output = self._resample(X, y)
        X = output[0]
        if len(output) == 2:
            y = output[1]

        # y_ = (
        #     label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        # )
        #
        # X_, y_ = arrays_transformer.transform(output[0], y_)

        if len(output) == 1:
            return X
        return X, y

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

        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        pass


