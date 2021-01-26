import numpy as np
from numpy import ndarray
import pandas as pd
from scipy import stats
import scipy.optimize as opt
from sklearn.base import BaseEstimator
from sklearn.cluster._kmeans import kmeans_plusplus
import matplotlib.pyplot as plt

from typing import List

from utils.simulate_returns import simulate_2state_gaussian
from models.hmm_cython import _log_forward_probs

''' TODO:

Add predict func -- USE Zuchinni
add stationary distribution - use it in sample function

Add method to get HMM params from a state sequence

'''


class BaseHiddenMarkov(BaseEstimator):
    """
    Base class for Hidden Markov methods with gaussian distributions.
    Contain methods related to:
    1. Initializing HMM parameters
    2. Methods that assumes an HMM is fitted and are used for sampling, prediction etc.


    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: int = 1e-6,
                 epochs: int = 1, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.max_iter = max_iter  # Max iterations to fit model
        self.epochs = epochs  # Set number of random inits used in model fitting
        self.tol = tol
        self.init = init

        # Init parameters initial distribution, transition matrix and state-dependent distributions from function
        np.random.seed(self.random_state)

    def _init_params(self, diag_uniform_dist: List[float] = [.7, .99]):  # TODO Would a tuple work?
        """
        Function to initialize HMM parameters.

        Parameters
        ----------
        diag_uniform_dist: 1D-array
            The lower and upper bounds of uniform distribution to sample init from.

        Returns
        -------
        self.T: N X N matrix of transition probabilities
        self.delta: 1 X N vector of initial probabilities
        self.mu: 1 X N vector of state dependent means
        self.std: 1 X N vector of state dependent STDs

        """

        if self.init == 'random':
            # Transition probabilities
            trans_prob = np.diag(np.random.uniform(low=diag_uniform_dist[0], high=diag_uniform_dist[1],
                                                   size=self.n_states))  # Sample diag as uniform dist
            remaining_row_nums = (1 - np.diag(trans_prob)) / (
                        self.n_states - 1)  # Spread the remaining mass evenly onto remaining row values
            trans_prob += remaining_row_nums.reshape(-1, 1)  # Add this to the uniform diagonal matrix
            np.fill_diagonal(trans_prob,
                             trans_prob.diagonal() - remaining_row_nums)  # And subtract these numbers from the diagonal so it remains uniform

            # initial distribution
            # init_dist = np.random.uniform(low=0.4, high=0.6, size=self.n_states)
            # init_dist /= np.sum(init_dist)
            init_dist = np.ones(self.n_states) / np.sum(self.n_states)  # initial distribution 1 X N vector

            # State dependent distributions
            mu = np.random.uniform(low=-0.05, high=0.15, size=self.n_states)  # np.random.rand(self.n_states)
            std = np.random.uniform(low=0.01, high=0.3, size=self.n_states)  # np.random.rand(self.n_states) #

        else:
            trans_prob = np.zeros((2, 2))
            trans_prob[0, 0] = 0.7
            trans_prob[0, 1] = 0.3
            trans_prob[1, 0] = 0.2
            trans_prob[1, 1] = 0.8
            init_dist = np.array([0.2, 0.8])  # initial distribution 1 X N vector
            mu = [-0.05, 0.1]
            std = [0.2, 0.1]

        self.T = trans_prob
        self.delta = init_dist
        self.mu = mu
        self.std = std

    def predict(self, X):
        state_preds, posteriors = self._viterbi(X)
        return state_preds, posteriors

    def emission_probs(self, X):
        """ Compute all different log probabilities p(x) given an observation sequence and n states

        Returns: T X N matrix
        """
        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init T X N matrix
        probs = np.zeros((T, self.n_states))

        # For all states evaluate the density function
        for j in range(self.n_states):
            log_probs[:, j] = stats.norm.logpdf(X, loc=self.mu[j], scale=self.std[j])

        probs = np.exp(log_probs)

        return probs, log_probs

    def _viterbi(self, X):
        """ Compute the most likely sequence of states given the observations
         To reduce CPU time consider storing each sequence --> will save T*m function evaluations

         """
        T = len(X)
        posteriors = np.zeros((T, self.n_states))  # Init T X N matrix
        self.emission_probs_, self.log_emission_probs_ = self.emission_probs(X)

        # Initiate posterior at time 0 and scale it as:
        posterior_temp = self.delta * self.emission_probs_[0, :]  # posteriors at time 0
        posteriors[0, :] = posterior_temp / np.sum(posterior_temp)  # Scaled posteriors at time 0

        # Do a forward recursion to compute posteriors
        for t in range(1, T):
            posterior_temp = np.max(posteriors[t - 1, :] * self.T, axis=1) * self.emission_probs_[t,
                                                                             :]  # TODO double check the max function returns the correct values
            posteriors[t, :] = posterior_temp / np.sum(posterior_temp)  # Scale rows to sum to 1

        # From posteriors get the the most likeley sequence of states i
        state_preds = np.zeros(T).astype(int)  # Vector of length N
        state_preds[-1] = np.argmax(posteriors[-1, :])  # Last most likely state is the index position

        # Do a backward recursion to calculate most likely state sequence
        for t in range(T - 2, -1, -1):  # Count backwards
            state_preds[t] = np.argmax(posteriors[t, :] * self.T[:, state_preds[
                                                                        t + 1]])  # TODO double check the max function returns the correct values

        return state_preds, posteriors

    def sample(self, n_samples: int):
        '''
        Sample from a fitted hmm.

        Parameters
        ----------
        n_samples: int
                Amount of samples to generate

        Returns
        -------
        Sample of same size n_samples
        '''
        state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=int)  # Array of possible states
        sample_states = np.zeros(n_samples).astype(int)  # Init sample vector
        sample_states[0] = np.random.choice(a=state_index, size=1, p=self.stationary_dist)  # First state is determined by initial dist

        for t in range(1, n_samples):
            # Each new state is chosen using the transition probs corresponding to the previous state sojourn.
            sample_states[t] = np.random.choice(a=state_index, size=1, p=self.T[sample_states[t - 1], :])

        samples = stats.norm.rvs(loc=self.mu[sample_states], scale=self.std[sample_states], size=n_samples)

        return samples, sample_states

    def get_hmm_params(self, X: ndarray, state_sequence: ndarray):  # TODO remove forward-looking params and slice X accordingly
        # Slice data
        if X.ndim == 1:  # Makes function compatible on Z
            X = X[(self.window_len - 1): -self.window_len]
        elif X.ndim > 1:
            X = X[:, 0]

        # group by states
        diff = np.diff(state_sequence)
        df_states = pd.DataFrame({'state_seq': state_sequence,
                                  'X': X,
                                  'state_sojourns': np.append([False], diff == 0),
                                  'state_changes': np.append([False], diff != 0)})

        state_groupby = df_states.groupby('state_seq')

        # Transition probabilities
        # TODO only works for a 2-state HMM
        self.T = np.diag(state_groupby['state_sojourns'].sum())
        state_changes = state_groupby['state_changes'].sum()
        self.T[0, 1] = state_changes[0]
        self.T[1, 0] = state_changes[1]
        self.T = self.T / self.T.sum(axis=1).reshape(-1, 1)  # make rows sum to 1

        # Conditional distributions
        self.mu = state_groupby['X'].mean().values.T  # transform mean back into 1darray
        self.std = state_groupby['X'].std().values.T

    def get_stationary_dist(self):
        ones = np.ones(shape=(self.n_states, self.n_states))
        identity = np.diag(ones)
        init_guess = np.ones(self.n_states) / self.n_states

        def solve_stationary(stationary_dist):
            return (stationary_dist @ (identity - self.T + ones)) - np.ones(self.n_states)

        stationary_dist = opt.root(solve_stationary, x0=init_guess)
        print(stationary_dist)

        self.stationary_dist = stationary_dist


if __name__ == '__main__':
    model = BaseHiddenMarkov(n_states=2)
    model._init_params()

    print(model.mu)
    print(model.std)
    print(model.T)
    print(model.delta)
