import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest
from DRSA import DRSA 
RANDOM_SEED = 42

@pytest.fixture
def decision_matrix():
    np.random.seed(RANDOM_SEED)
    n_attr = 3
    n_alt = 20
    dm = pd.DataFrame(np.random.sample(size=(n_alt, n_attr)), columns=['attr'+str(item) for item in range(n_attr)])
    target = pd.DataFrame(np.random.randint(0, 2, n_alt), columns=['target'])
    dm = pd.concat([dm, target], axis=1)
    return dm

def test_classifier(decision_matrix):
    X = decision_matrix.iloc[:, :-1]
    y = decision_matrix.iloc[:, -1]
    XX = pd.DataFrame()
    for x in X:
        XX[x+"-"] = X[x].copy()
    X = pd.concat([X, XX], axis=1)
    test_size = 5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    drsa = DRSA([1, 1, 1, -1, -1, -1,])
    drsa.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
    y_pred = drsa.predict(X_test)
    assert((np.array([0, 1, 0, 0, 0]) == y_pred).sum() == test_size)
