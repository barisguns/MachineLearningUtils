import numpy as np
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from mldsutils.preprocessing import manual_range_column_select, ManualRangeVariableSelector


# Test manual_range_column_select function
def test_manual_range_column_select():
    # Test case 1: Single-row matrix
    X1 = np.array([[1, 2, 3, 4, 5]])
    expected_result1 = np.array([[2, 3]])
    assert np.array_equal(manual_range_column_select(X1, [(1, 3)]), expected_result1)

    # Test case 2: Multi-row matrix
    X2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    expected_result2 = np.array([[2, 3], [7, 8]])
    assert np.array_equal(manual_range_column_select(X2, [(1, 3)]), expected_result2)

    # Test case 3: Empty matrix
    X3 = np.array([])
    expected_result3 = np.array([])
    assert np.array_equal(manual_range_column_select(X3, [(1, 3)]), expected_result3)


# Test ManualRangeVariableSelector transformer
def test_manual_range_variable_selector():
    # Generate synthetic data for testing
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)

    # Define the variable range list for selection
    variable_range_list = [(1, 3), (7, 9)]

    # Create the transformer pipeline
    transformer = Pipeline([
        ('manual_selector', ManualRangeVariableSelector(variable_range_list))
    ])

    # Apply the transformer
    X_transformed = transformer.transform(X)

    # Check the transformed shape and content
    assert X_transformed.shape == (100, 4)
    assert np.array_equal(X_transformed, X[:, [1, 2, 7, 8]])

