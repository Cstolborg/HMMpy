import numpy as np
from scipy import stats
from scipy.special import digamma
import scipy.optimize as opt
import matplotlib.pyplot as plt

from typing import List

from utils import simulate_2state_gaussian
from base_hmm import BaseHiddenMarkov


class GradientMarkov(BaseHiddenMarkov):

    def __init__(self, n_states: int = 2, params_init: str = 'random', max_iter: int = 100, tol: int = 1e-4,
                 epochs: int = 1, random_state: int = 42):
        super().__init__(n_states, params_init, max_iter, tol, epochs, random_state)

        # Random init of state distributions
        np.random.seed(self.random_state)




if __name__ == '__main__':

    model = GradientMarkov(n_states=2)

