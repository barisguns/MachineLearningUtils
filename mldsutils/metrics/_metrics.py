from sklearn.metrics._scorer import _SCORERS

_METRICS = {}

for i in _SCORERS:
    _METRICS[i] = (_SCORERS[i]._score_func, _SCORERS[i]._sign, _SCORERS[i]._kwargs)

