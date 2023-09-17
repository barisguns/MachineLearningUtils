import numpy as np
from copy import deepcopy


def get_retained_y_test(estimator, y_test):
    y_test_new = deepcopy(y_test)
    if hasattr(estimator, "indices_retained") and \
            estimator.indices_retained is not None and len(estimator.indices_retained) != 0:

        if isinstance(estimator.indices_retained[0], (list, np.ndarray)):
            for i in estimator.indices_retained:
                y_test_new = [y_test[x] for x in i]
        else:
            y_test_new = [y_test[x] for x in estimator.indices_retained]
    return y_test_new
