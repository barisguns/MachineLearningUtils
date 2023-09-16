from sklearn.metrics._scorer import positive_likelihood_ratio, negative_likelihood_ratio

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    explained_variance_score,
    f1_score,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    v_measure_score,
)

_METRICS = dict(
    explained_variance=explained_variance_score,
    r2=r2_score,
    max_error=max_error,
    matthews_corrcoef=matthews_corrcoef,
    neg_median_absolute_error=median_absolute_error,
    neg_mean_absolute_error=mean_absolute_error,
    neg_mean_absolute_percentage_error=mean_absolute_percentage_error,  # noqa
    neg_mean_squared_error=mean_squared_error,
    neg_mean_squared_log_error=mean_squared_log_error,
    neg_root_mean_squared_error=(mean_squared_error, {"squared": True}),
    neg_mean_poisson_deviance=mean_poisson_deviance,
    neg_mean_gamma_deviance=mean_gamma_deviance,
    accuracy=accuracy_score,
    top_k_accuracy=top_k_accuracy_score,
    roc_auc=roc_auc_score,
    # TODO: AFTER IMPLEMENTING PREDICT PROBA IN PIPELINE, COMMENTED OUT SCORERS SHOULD BE MADE AVAILABLE
    # roc_auc_ovr=roc_auc_ovr_scorer,
    # roc_auc_ovo=roc_auc_ovo_scorer,
    # roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer,
    # roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer,
    balanced_accuracy=balanced_accuracy_score,
    average_precision=average_precision_score,
    neg_log_loss=log_loss,
    neg_brier_score=brier_score_loss,
    positive_likelihood_ratio=positive_likelihood_ratio,
    neg_negative_likelihood_ratio=negative_likelihood_ratio,
    # Cluster metrics that use supervised evaluation
    adjusted_rand_score=adjusted_rand_score,
    rand_score=rand_score,
    homogeneity_score=homogeneity_score,
    completeness_score=completeness_score,
    v_measure_score=v_measure_score,
    mutual_info_score=mutual_info_score,
    adjusted_mutual_info_score=adjusted_mutual_info_score,
    normalized_mutual_info_score=normalized_mutual_info_score,
    fowlkes_mallows_score=fowlkes_mallows_score,
)

for name, metric in [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
    ("jaccard", jaccard_score),
]:
    _METRICS[name] = (metric, {"average": "binary"})
    for average in ["macro", "micro", "samples", "weighted"]:
        qualified_name = "{0}_{1}".format(name, average)
        _METRICS[qualified_name] = (metric, {"pos_label": None, "average": average})


def get_scorer_names():
    """Get the names of all available scorers.

    These names can be passed to :func:`~sklearn.metrics.get_scorer` to
    retrieve the scorer object.

    Returns
    -------
    list of str
        Names of all available scorers.
    """
    return sorted(_METRICS.keys())

