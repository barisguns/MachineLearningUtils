from sklearn.metrics import confusion_matrix, f1_score
from mldsutils.model_selection import get_retained_y_test


def confusion_matrix_scorer(clf, X, y):
    """
    This custom scorer follows sklearn scorer signature and takes classifier, data matrix and true labels as input.
    It returns a dictionary that includes confusion matrix components which are accessed by
     calling sklearn.metrics.confusion_matrix().
    :param clf: A sklearn classifier object.
    :param X: Data matrix.
    :param y: True labels.
    :return:
    """
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}


def true_positive_scorer(clf, X, y):
    """
    This custom scorer follows sklearn scorer signature and takes classifier, data matrix and true labels as input.
    It returns the number of true positives which is accessed by calling sklearn.metrics.confusion_matrix().
    :param clf:
    :param X:
    :param y:
    :return:
    """
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[1, 1]


def true_negative_scorer(clf, X, y):
    """
    This custom scorer follows sklearn scorer signature and takes classifier, data matrix and true labels as input.
    It returns the number of true negatives which is accessed by calling sklearn.metrics.confusion_matrix().
    :param clf:
    :param X:
    :param y:
    :return:
    """
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[0, 0]


def false_negative_scorer(clf, X, y):
    """
    This custom scorer follows sklearn scorer signature and takes classifier, data matrix and true labels as input.
    It returns the number of false negatives which is accessed by calling sklearn.metrics.confusion_matrix().
    :param clf:
    :param X:
    :param y:
    :return:
    """
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[1, 0]


def false_positive_scorer(clf, X, y):
    """
    This custom scorer follows sklearn scorer signature and takes classifier, data matrix and true labels as input.
    It returns the number of false positives which is accessed by calling sklearn.metrics.confusion_matrix().
    :param clf:
    :param X:
    :param y:
    :return:
    """
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[0, 1]


def outlier_f1_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_test = get_retained_y_test(estimator, y)
    score = f1_score(y_test, y_pred)
    return score
