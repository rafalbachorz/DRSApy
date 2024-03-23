import numpy as np
import pandas as pd
RANDOM_SEED = 42

def test_classifier():
    np.random.seed(RANDOM_SEED)
    n_attr = 3
    n_alt = 10
    dm = pd.DataFrame(np.random.sample(size=(n_alt, n_attr)), columns=['attr'+str(item) for item in range(n_attr)])
    target = pd.DataFrame(np.random.randint(0, 2, n_alt), columns=['target'])
    dm = pd.concat([dm, target], axis=1)
    pass
