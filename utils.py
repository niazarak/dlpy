import random

import numpy as np


def one_hot(x, classes):
    res = np.zeros(shape=(x.shape[0], classes))
    res[np.arange(x.shape[0]), x] = 1
    return res


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
