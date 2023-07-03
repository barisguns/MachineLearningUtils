import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mldsutils.pipeline import Pipeline
from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import skorch
from sklearn.utils.validation import check_is_fitted


class NumpyPytorchDatatypeResampler(SamplerMixin, BaseEstimator):
    """
    This is a custom imblearn resampler which transforms data into formats that would work as PyTorch inputs.
    It is especially useful when outputs of sklearn/imblearn transformers are to be given to PyTorch models as input.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        return X.astype(np.float32), y.astype(np.int64)

    def transform(self, X, y=None):
        return X.astype(np.float32), None, None

    def _fit_resample(self, X, y):
        return X, y


# Custom PyTorch model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture here
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_custom_resampler():
    # Create a synthetic imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20, weights=[0.9, 0.1], random_state=42)

    # Define the sklearn transformer
    scaler = StandardScaler()

    # Define your custom resampler
    resampler = NumpyPytorchDatatypeResampler()
    # Make sure fit function of the resampler returns self
    assert resampler == resampler.fit(X, y)

    # Create the PyTorch model
    model = MyModel()

    X_sc = scaler.fit_transform(X)

    X_transformed, _, _ = resampler.transform(X_sc)
    # Check if the transformed data is in the expected input format for torch models
    assert X_transformed.dtype == np.float32

    X_resampled, y_resampled = resampler.fit_resample(X_sc, y)

    # Check if the resampled data is in the expected input format for torch models
    assert X_resampled.dtype == np.float32
    assert y_resampled.dtype == np.int64

    # Define the pipeline
    pipeline = Pipeline([("s", scaler), ("res", resampler),
                         ('nn', skorch.NeuralNetClassifier(model))])

    # Train the model using the pipeline
    pipeline.fit(X, y)

    # Predict the trained data
    y_pred = pipeline.predict(X)

    # Check if the model is fitted
    assert hasattr(pipeline, "classes_")
    # Check if the length of the output is the same as the input
    assert len(y_pred) == np.shape(X)[0]

