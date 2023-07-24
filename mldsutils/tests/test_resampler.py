import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from mldsutils.pipeline import Pipeline
from mldsutils.preprocessing import NumpyPytorchDatatypeResampler
import torch
import torch.nn as nn
import skorch


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

    # Define the custom resampler
    resampler = NumpyPytorchDatatypeResampler()

    # Create the PyTorch model
    model = MyModel()

    X_sc = scaler.fit_transform(X)

    X_transformed = resampler.resample(X_sc)[0]
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

