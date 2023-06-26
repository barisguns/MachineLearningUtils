from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import check_array
from mldsutils.metrics import (
    confusion_matrix_scorer,
    true_positive_scorer,
    true_negative_scorer,
    false_negative_scorer,
    false_positive_scorer,
)
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
import numpy as np


# Mock classifier object
class MockClassifier:
    def __init__(self, y_pred):
        self.y_pred = y_pred

    def predict(self, X):
        return self.y_pred


y_true = [0, 1, 0, 1]
y_pred = [0, 1, 1, 1]
cm = confusion_matrix(y_true, y_pred)


# Test if confusion_matrix_scorer is agreeable with sklearn.confusion_matrix()
# and it satisfies sklearn.cross_validate() "scoring" parameter expectations.
def test_confusion_matrix_scorer():
    clf = MockClassifier(y_pred)
    X = check_array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_scores = {
        'tn': cm[0, 0],
        'fp': cm[0, 1],
        'fn': cm[1, 0],
        'tp': cm[1, 1]
    }
    score = confusion_matrix_scorer(clf, X, y_true)
    assert isinstance(score, dict)
    assert confusion_matrix_scorer(clf, X, y_true) == expected_scores


# Test if true_positive_scorer is agreeable with sklearn.confusion_matrix()
# and it satisfies sklearn.cross_validate() "scoring" parameter expectations.
def test_true_positive_scorer():
    clf = MockClassifier(y_pred)
    X = check_array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_score = cm[1, 1]

    score = true_positive_scorer(clf, X, y_true)

    assert isinstance(score, (int, np.integer))
    assert score == expected_score


# Test if true_negative_scorer is agreeable with sklearn.confusion_matrix()
# and it satisfies sklearn.cross_validate() "scoring" parameter expectations.
def test_true_negative_scorer():
    clf = MockClassifier(y_pred)
    X = check_array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_score = cm[0, 0]

    score = true_negative_scorer(clf, X, y_true)

    assert isinstance(score, (int, np.integer))
    assert score == expected_score


# Test if false_negative_scorer is agreeable with sklearn.confusion_matrix()
# and it satisfies sklearn.cross_validate() "scoring" parameter expectations.
def test_false_negative_scorer():
    clf = MockClassifier(y_pred)
    X = check_array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_score = cm[1, 0]

    score = false_negative_scorer(clf, X, y_true)

    assert isinstance(score, (int, np.integer))
    assert score == expected_score


# Test if false_positive_scorer is agreeable with sklearn.confusion_matrix()
def test_false_positive_scorer():
    clf = MockClassifier(y_pred)
    X = check_array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_score = cm[0, 1]

    score = false_positive_scorer(clf, X, y_true)

    assert isinstance(score, (int, np.integer))
    assert score == expected_score


# Generate synthetic data for testing
X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# Create a dummy classifier
clf = DummyClassifier(strategy="most_frequent")

# Define the cross-validation folds
cv = 5


# Test confusion_matrix_scorer within cross-validation´
def test_confusion_matrix_scorer_cross_val():
    scores = cross_validate(clf, X, y, scoring=confusion_matrix_scorer, cv=cv)
    # Ensure the scores dictionary has correct components in the correct type

    assert "test_tn" in scores
    assert "test_fn" in scores
    assert "test_tp" in scores
    assert "test_fp" in scores
    assert isinstance(scores["test_tn"], np.ndarray)
    assert isinstance(scores["test_fn"], np.ndarray)
    assert isinstance(scores["test_tp"], np.ndarray)
    assert isinstance(scores["test_fp"], np.ndarray)
    # TODO: check if all the elements of the arrays are integers and their total is equivalent to number of data points


# Test true_positive_scorer within cross-validation
def test_true_positive_scorer_cross_val():
    scores = cross_val_score(clf, X, y, scoring=true_positive_scorer, cv=cv)
    # Ensure the scores are integers
    for score in scores:
        assert isinstance(score, (int, np.integer))


# Test true_negative_scorer within cross-validation
def test_true_negative_scorer_cross_val():
    scores = cross_val_score(clf, X, y, scoring=true_negative_scorer, cv=cv)
    # Ensure the scores are integers
    for score in scores:
        assert isinstance(score, (int, np.integer))


# Test false_negative_scorer within cross-validation
def test_false_negative_scorer_cross_val():
    scores = cross_val_score(clf, X, y, scoring=false_negative_scorer, cv=cv)
    # Ensure the scores are integers
    for score in scores:
        assert isinstance(score, (int, np.integer))


# Test false_positive_scorer within cross-validation
def test_false_positive_scorer_cross_val():
    scores = cross_val_score(clf, X, y, scoring=false_positive_scorer, cv=cv)
    # Ensure the scores are integers
    for score in scores:
        assert isinstance(score, (int, np.integer))

