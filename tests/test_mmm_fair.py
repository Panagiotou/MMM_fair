import numpy as np
from mmm_fair import MMM_Fair

def test_fit_and_predict():
    X = np.array([[1,2],[2,3],[3,4],[4,5]])
    y = np.array([0, 1, 0, 1])
    saIndex = np.array([[0],[1],[1],[0]])  # example sensitive attribute index
    saValue = {"sensitive_attr": 0}        # example threshold or group

    clf = MMM_Fair(saIndex=saIndex, saValue=saValue)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)