# Author: David Burns
# License: BSD


import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from seglearn.datasets import load_watch
from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, Segment

# load the data
data = load_watch()
X = data['X']
y = data['y']

# create a feature representation pipeline
steps = [('seg', Segment()),
         ('features', FeatureRep()),
         ('scaler', StandardScaler()),
         ('rf', RandomForestClassifier(n_estimators=20))]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scorer = make_scorer(f1_score, average='macro')
pipe = Pype(steps, scorer=scorer)
cv_scores = cross_validate(pipe, X, y, cv=4, return_train_score=True)
print("CV F1 Scores: ", pd.DataFrame(cv_scores))

