"""
This script demonstrates the usability of the custom sklearn transformer ManualRangeVariableSelector
from mldsutils.preprocessing. It uses the transformer in a pipeline, combines it with a regressor and
fits and predicts using the pipeline.
"""

# TODO: visualise the X somehow while it is used in the pipeline.fit(), before and after the variable selection.

from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mldsutils.preprocessing import ManualRangeVariableSelector

# Load the Iris dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the desired feature ranges
feature_ranges = [(1, 2), (3, 3)]

# Create the pipeline with the ManualVariableRangeSelector and LDA
pipeline = make_pipeline(
    ManualRangeVariableSelector(feature_ranges),
    LogisticRegression()
)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = pipeline.predict(X_test)

# Calculate the accuracy of the predictions
r2_score = r2_score(y_test, y_pred)

print("R2_score:", r2_score)

