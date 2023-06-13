from sklearn.base import BaseEstimator, TransformerMixin


def manual_range_column_select(X, variable_range_list):
    """
    Takes a 2d matrix and a list of tuples, eliminates all the columns from the matrix
    whose indices does not fall between the values in any of the tuples in the list.

    :param X: 2d matrix.
    :param variable_range_list: list of tuples that contain the index values where the columns with the indices
    between those values will be kept and the rest will be filtered out.
    :return:
    """
    var_lst = []
    for i in variable_range_list:
        var_lst = var_lst + list(range(i[0], i[1]))
    if X.shape[0] == 1:
        X = X[0][var_lst].reshape(1, -1)
    else:
        X = X[:, var_lst]
    return X


class ManualRangeVariableSelector(BaseEstimator, TransformerMixin):
    """
    This is a custom sklearn transformer class for manual variable selection.
    Input is a data matrix (X) and
    a list of tuples which contain start and end indices of the variables to be selected within the range:
    All the variables that does not have indices
    that fall between any of those (start:end) couples will be filtered out from the data matrix.

    Example input: [(3,5), (7,9)]
    """
    def __init__(self, variable_range_list):
        """
        Constructs a manual variable selector object.
        :param variable_range_list:
        """
        self.variable_range_list = variable_range_list
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Takes data matrix and the variable range list and returns the filtered data matrix.
        :param X:
        :param y:
        :return:
        """
        return manual_range_column_select(X, self.variable_range_list)
