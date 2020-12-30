import os
import random

import numpy as np


def pytest_configure(config):
    _random_seed = int(os.environ.get("HMMLEARN_SEED",
                                      np.random.uniform() * (2**31 - 1)))
    print("set RNG seed to {}".format(_random_seed))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
