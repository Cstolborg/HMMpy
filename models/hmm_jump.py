import numpy as np
from numpy import ndarray
from scipy import stats
import scipy.optimize as opt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster._kmeans import kmeans_plusplus

import matplotlib.pyplot as plt

from utils.simulate_returns import simulate_2state_gaussian
from models.hmm_base import BaseHiddenMarkov


''' TODO:

Add predict func -- USE Zuchinni
add stationary distribution
 
implement simulation and BAC to choose jump penalty.

Z-score standardisation

'''

class JumpHMM(BaseHiddenMarkov):

    def __init__(self, n_states: int = 2, jump_penalty: float = .2, init: str = 'kmeans++',
                 max_iter: int = 30, tol: int = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

        # Init parameters initial distribution, transition matrix and state-dependent distributions
        self.jump_penalty = jump_penalty

        self.best_objective_likelihood = np.inf
        self.old_objective_likelihood = np.inf

    def construct_features(self, X: ndarray, window_len: int):  # TODO remove forward-looking params and slice X accordingly
        N = len(X)
        df = pd.DataFrame(X)

        df['Left local mean'] = df.rolling(window_len).mean()
        df['Left local std'] = df[0].rolling(window_len).std(ddof=0)

        df['Right local mean'] = df[0].rolling(window_len).mean().shift(-window_len+1)
        df['Right local std'] = df[0].rolling(window_len).std(ddof=0).shift(-window_len + 1)

        look_ahead = df[0].rolling(window_len).sum().shift(-window_len)  # Looks forward with window_len (Helper 1)
        look_back = df[0].rolling(window_len).sum()  # Includes current position and looks window_len - 1 backward (Helper 2)
        df['Central local mean'] = (look_ahead + look_back) / (2 * window_len)
        df['Centered local std'] = df[0].rolling(window_len * 2).std(ddof=0).shift(-window_len)  # Rolls from 0 and 2x length iteratively, then shifts back 1x window length

        # Absolute changes
        #Y[1:, 1] = np.abs(np.diff(X))
        #Y[:-1, 2] = np.abs(np.diff(X))
        Z = df.dropna().to_numpy()
        self.n_features = Z.shape[1]
        self.window_len = window_len

        return Z

    def _fit_theta(self, Z):
        for j in range(self.n_states):
            state_slicer = self.state_seq == j  # Boolean array of True/False

            N_j = sum(state_slicer)  # Number of terms in state j
            z = Z[state_slicer].sum(axis=0)  # Sum each feature across time
            self.theta[:, j] = 1 / N_j * z

    def fit(self, Z: ndarray, verbose=0):
        # init params
        self._init_params(Z, output_hmm_params=False)
        self.old_state_seq = self.state_seq  # Required in fit() method

        for epoch in range(self.epochs):
            # Do new init at each epoch
            if epoch > 0: self._init_params(Z, output_hmm_params=False)

            for iter in range(self.max_iter):
                self._fit_theta(Z)
                self._fit_state_seq(Z)

                # Check convergence criterion
                crit1 = np.array_equal(self.state_seq, self.old_state_seq)  # No change in state sequence
                crit2 = np.abs(self.old_objective_likelihood - self.objective_likelihood)
                if crit1 == True or crit2 < self.tol:
                    # If model is converged check if current epoch is the best
                    if self.objective_likelihood < self.best_objective_likelihood:
                        self.best_objective_likelihood = self.objective_likelihood
                        self.best_state_seq = self.state_seq
                        self.best_theta = self.theta
                        self.get_hmm_params(Z, state_sequence=self.best_state_seq)

                    print(f'Epoch {epoch} -- Iter {iter} -- likelihood {self.objective_likelihood} -- Theta {self.theta[0]} ')#Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')
                    break

                elif iter == self.max_iter - 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_state_seq = self.state_seq
                    self.old_objective_likelihood = self.objective_likelihood


if __name__ == '__main__':
    model = JumpHMM(n_states=2, random_state=42)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions

    Z = model.construct_features(returns, window_len=6)

    model.fit(Z)
    #print(model.T)
    model.stationary_dist = np.array([0.5, 0.5])
    print(model.sample(10))

    plotting = False
    if plotting == True:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        #ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
        #ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
        #ax[1].plot(first_seq, label='First Predicted states', ls='dashdot')
        ax[1].plot(model.best_state_seq, label='Predicted states', ls='dotted')
        ax[1].plot(true_regimes, label='True states', ls='dashed')

        plt.legend()
        plt.show()


