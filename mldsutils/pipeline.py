"""
The :mod:`imblearn.pipeline` module implements utilities to build a
composite estimator, as a chain of transforms, samples and estimators.
"""
# Adapted from scikit-learn

# Author: Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
#         Christos Aridas
#         Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: BSD
from sklearn.utils.metaestimators import available_if
from imblearn import pipeline
from sklearn import pipeline as skpipe
from imblearn.base import _ParamsValidationMixin
from imblearn.utils._param_validation import HasMethods, validate_params

__all__ = ["Pipeline"]


class Pipeline(pipeline.Pipeline):
    """Pipeline of transforms and resamples with a final estimator.

    Sequentially apply a list of transforms, sampling, and a final estimator.
    Intermediate steps of the pipeline must be transformers or resamplers,
    that is, they must implement fit, transform and sample methods.
    The samplers are only applied during fit.
    The final estimator only needs to implement fit.
    The transformers and samplers in the pipeline can be cached using
    ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None``.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing
        fit/transform/fit_resample) that are chained, in the order in which
        they are chained, with the last object an estimator.

    memory : Instance of joblib.Memory or str, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during first step `fit` method.

    See Also
    --------
    make_pipeline : Helper function to make pipeline.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py`

    .. warning::
       A surprising behaviour of the `imbalanced-learn` pipeline is that it
       breaks the `scikit-learn` contract where one expects
       `estimmator.fit_transform(X, y)` to be equivalent to
       `estimator.fit(X, y).transform(X)`.

       The semantic of `fit_resample` is to be applied only during the fit
       stage. Therefore, resampling will happen when calling `fit_transform`
       while it will only happen on the `fit` stage when calling `fit` and
       `transform` separately. Practically, `fit_transform` will lead to a
       resampled dataset while `fit` and `transform` will not.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split as tts
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.neighbors import KNeighborsClassifier as KNN
    >>> from sklearn.metrics import classification_report
    >>> from imblearn.over_sampling import SMOTE
    >>> from imblearn.pipeline import Pipeline
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print(f'Original dataset shape {Counter(y)}')
    Original dataset shape Counter({1: 900, 0: 100})
    >>> pca = PCA()
    >>> smt = SMOTE(random_state=42)
    >>> knn = KNN()
    >>> pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])
    >>> X_train, X_test, y_train, y_test = tts(X, y, random_state=42)
    >>> pipeline.fit(X_train, y_train)
    Pipeline(...)
    >>> y_hat = pipeline.predict(X_test)
    >>> print(classification_report(y_test, y_hat))
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.87      1.00      0.93        26
               1       1.00      0.98      0.99       224
    <BLANKLINE>
        accuracy                           0.98       250
       macro avg       0.93      0.99      0.96       250
    weighted avg       0.99      0.98      0.98       250
    <BLANKLINE>
    """

    _parameter_constraints: dict = {
        "steps": "no_validation",  # validated in `_validate_steps`
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
    }

    # BaseEstimator interface

    def __init__(self, steps, *, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.indices_retained = []

    # Estimator interface

    @available_if(skpipe._final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False, filter_passthrough=False, filter_resample=False):
            if hasattr(transform, "_resample"):
                Xt = transform.resample(Xt)
                self.indices_retained.append(transform.indices_retained)
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][1].predict(Xt, **predict_params)


@validate_params(
    {"memory": [None, str, HasMethods(["cache"])], "verbose": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def make_pipeline(*steps, memory=None, verbose=False):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators
        A list of estimators.

    memory : None, str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : Pipeline
        Returns an imbalanced-learn `Pipeline` instance that handles samplers.

    See Also
    --------
    imblearn.pipeline.Pipeline : Class for creating a pipeline of
        transforms with a final estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    return Pipeline(skpipe._name_estimators(steps), memory=memory, verbose=verbose)
