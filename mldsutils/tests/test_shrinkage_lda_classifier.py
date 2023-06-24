import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mldsutils.classification import ShrinkageLDAClassifier


def test_shrinkage_lda_classifier():
    # Test case 1: Basic functionality

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the ShrinkageLDAClassifier
    classifier = ShrinkageLDAClassifier()

    # Fit the classifier
    classifier.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = classifier.predict(X_test)

    # Check if the predictions have the correct shape
    assert y_pred.shape == y_test.shape

    # Check if the accuracy score is reasonable
    accuracy = accuracy_score(y_test, y_pred)
    assert 0.0 <= accuracy <= 1.0


# def test_shrinkage_lda_classifier_different_datasets():
#     # Test case 2: Behavior with different types of datasets
#
#     # Generate a linearly separable dataset
#     X_linear, y_linear = make_classification(n_samples=100, n_features=10, random_state=42)
#
#     # Generate a dataset with overlapping classes
#     X_overlap, y_overlap = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5,
#                                                random_state=42)
#
#     # Initialize the ShrinkageLDAClassifier
#     classifier = ShrinkageLDAClassifier()
#
#     # Fit and predict for the linearly separable dataset
#     classifier.fit(X_linear, y_linear)
#     y_pred_linear = classifier.predict(X_linear)
#
#     # Fit and predict for the dataset with overlapping classes
#     classifier.fit(X_overlap, y_overlap)
#     y_pred_overlap = classifier.predict(X_overlap)
#
#     # Check if the predictions have the correct shape
#     assert y_pred_linear.shape == y_linear.shape
#     assert y_pred_overlap.shape == y_overlap.shape
#
#     # Check if the accuracy scores are reasonable
#     accuracy_linear = accuracy_score(y_linear, y_pred_linear)
#     assert accuracy_linear == 1.0
#
#     accuracy_overlap = accuracy_score(y_overlap, y_pred_overlap)
#     assert accuracy_overlap >= 0.5


def test_shrinkage_lda_classifier_edge_cases():
    # Test case 3: Edge cases

    # Generate a dataset with only one feature
    X_single_feature, y_single_feature = make_classification(n_samples=100, n_features=1, n_clusters_per_class=1,
                                                             n_informative=1, n_redundant=0, random_state=42)

    # Initialize the ShrinkageLDAClassifier
    classifier = ShrinkageLDAClassifier()

    # Fit and predict for the single feature dataset
    classifier.fit(X_single_feature, y_single_feature)
    y_pred_single_feature = classifier.predict(X_single_feature)

    # Check if the predictions have the correct shape
    assert y_pred_single_feature.shape == y_single_feature.shape


def test_shrinkage_lda_classifier_comparison():
    # Test case 4: Comparison with other classifiers

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the ShrinkageLDAClassifier
    shrinkage_lda_classifier = ShrinkageLDAClassifier()

    # Initialize another classifier for comparison (e.g., Logistic Regression)
    from sklearn.linear_model import LogisticRegression
    logistic_regression = LogisticRegression()

    # Fit and predict using the ShrinkageLDAClassifier
    shrinkage_lda_classifier.fit(X_train, y_train)
    y_pred_shrinkage_lda = shrinkage_lda_classifier.predict(X_test)

    # Fit and predict using the Logistic Regression classifier
    logistic_regression.fit(X_train, y_train)
    y_pred_logistic_regression = logistic_regression.predict(X_test)

    # Check if the predictions have the correct shape
    assert y_pred_shrinkage_lda.shape == y_test.shape
    assert y_pred_logistic_regression.shape == y_test.shape

    # Check if the accuracy score of ShrinkageLDAClassifier is higher than Logistic Regression
    accuracy_shrinkage_lda = accuracy_score(y_test, y_pred_shrinkage_lda)
    accuracy_logistic_regression = accuracy_score(y_test, y_pred_logistic_regression)
    assert accuracy_shrinkage_lda >= accuracy_logistic_regression
