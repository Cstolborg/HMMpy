import numpy as np
from numpy import ndarray
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
import scipy.optimize as opt
from sklearn.base import BaseEstimator
from sklearn.cluster._kmeans import kmeans_plusplus
import matplotlib.pyplot as plt


from utils.simulate_returns import simulate_2state_gaussian

import pyximport; pyximport.install()  # TODO can only be active during development -- must be done through setup.py
from models import hmm_cython

''' TODO:

Very bad integration of kmeans++ init for EM algos.
 -   Consider moving entirely into jump models
 -   OR make a better version


add stationary distribution - use it in sample function

'''


class BaseHiddenMarkov(BaseEstimator):
    """
    Parent class for Hidden Markov methods with gaussian distributions.
    Contain methods related to:
    1. Initializing HMM parameters
    2. Methods that assumes an HMM is fitted and are used for sampling, prediction etc.

    To fit HMMs refer to the respective child classes

    Parameters
    ----------
    n_states : int, default=2
            Number of hidden states
    max_iter : int, default=100
            Maximum number of iterations to perform during expectation-maximization
    tol : float, default=1e-6
            Criterion for early stopping
    epochs : int, default=1
            Number of complete passes through the data to improve fit
    random_state : int, default = 42
            Parameter set to recreate output
    init : str
            Set to 'random' for random initialization.
            Set to None for deterministic init.
    Attributes
    ----------
    mu : ndarray of shape (n_states,)
        Fitted means for each state
    std : ndarray of shape (n_states,)
        Fitted std for each state
    T : ndarray of shape (n_states, n_states)
        Matrix of transition probabilities between states
    delta : ndarray of shape (n_states,)
        Initial state occupation distribution
    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: float = 1e-6,
                 epochs: int = 1, random_state: int = 42):
        self.window_len = None
        self.n_states = n_states
        self.random_state = random_state
        self.max_iter = max_iter  # Max iterations to fit model
        self.epochs = epochs  # Set number of random inits used in model fitting
        self.tol = tol
        self.init = init

        self.jump_penalty = 0.2  # TODO figure out better integration between kmeans++ init and jump

        # Init parameters initial distribution, transition matrix and state-dependent distributions from function
        np.random.seed(self.random_state)

    def _init_params(self, X=None, diag_uniform_dist = (.7, .99), output_hmm_params=True):
        """
        Function to initialize HMM parameters. Can do so using kmeans++, randomly
        or completely deterministic.

        Parameters
        ----------
        diag_uniform_dist: 1D-array
            The lower and upper bounds of uniform distribution to sample init from.

        Attributes
        -------
        self.T: N X N matrix of transition probabilities
        self.delta: 1 X N vector of initial probabilities
        self.mu: 1 X N vector of state dependent means
        self.std: 1 X N vector of state dependent STDs

        """

        if self.init == "kmeans++":
            if not isinstance(X, np.ndarray):
                raise Exception("To initialize with kmeans++ a sequence of data must be provided in an ndarray")
            if X.ndim == 1:
                X = X.reshape(-1, 1)  # Compatible with sklearn

            # Theta - only used in jump models and are discarded in EM models
            centroids, _ = kmeans_plusplus(X, self.n_states)  # Use sklearns kmeans++ algorithm
            self.theta = centroids.T  # Transpose to get shape: N_features X N_states

            # State sequence
            self._fit_state_seq(X)  # this function implicitly updates self.state_seq

            if output_hmm_params == True: # output all hmm parameters from state sequence
                self.get_params_from_seq(X, state_sequence=self.state_seq)

        elif self.init == 'random':
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

            self.T = trans_prob
            self.delta = init_dist
            self.mu = mu
            self.std = std

        else:
            # Init theta as zeros and sample state seq from uniform dist
            self.theta = np.zeros(shape=(self.n_features, self.n_states))  # Init as empty matrix
            state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=int)  # Array of possible states
            state_seq = np.random.choice(a=state_index, size=len(X))  # Sample sequence uniformly
            self.state_seq = state_seq

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

    def _fit_state_seq(self, X: ndarray):
        T = len(X)

        losses = np.zeros(shape=(T, self.n_states))  # Init T X N matrix
        losses[-1, :] = self._l2_norm_squared(X[-1], self.theta)  # loss corresponding to last state

        # Do a backward recursion to get losses
        for t in range(T - 2, -1, -1):  # Count backwards
            current_loss = self._l2_norm_squared(X[t], self.theta)  # n-state vector of current losses
            last_loss = self._l2_norm_squared(X[t + 1], self.theta)  # n-state vector of last losses

            for j in range(self.n_states):  # TODO redefine this as a matrix problem and get rid of loop
                state_change_penalty = np.ones(self.n_states) * self.jump_penalty  # Init jump penalty
                state_change_penalty[j] = 0  # And remove it for all but current state
                losses[t, j] = current_loss[j] + np.min(last_loss + state_change_penalty)

        # From losses get the most likely sequence of states i
        state_preds = np.zeros(T).astype(int)  # Vector of length N
        state_preds[0] = np.argmin(losses[0])  # First most likely state is the index position

        # Do a forward recursion to calculate most likely state sequence
        for t in range(1, T):  # Count backwards
            last_state = state_preds[t-1]

            state_change_penalty = np.ones(self.n_states) * self.jump_penalty  # Init jump penalty
            state_change_penalty[last_state] = 0  # And remove it for all but current state

            state_preds[t] = np.argmin(losses[t] + state_change_penalty)

        # Finally compute score of objective function
        all_likelihoods = losses[np.arange(len(losses)), state_preds].sum()
        state_changes = np.diff(state_preds) != 0   # True/False array showing state changes
        jump_penalty = (state_changes * self.jump_penalty).sum()  # Multiply all True values with penalty

        self.objective_likelihood = all_likelihoods + jump_penalty  # Float
        self.state_seq = state_preds

    def _log_forward_proba(self):
        """
        Compute log forward probabilities in scaled form.
        Forward probability is essentially the joint probability of observing
        a state = i and observation sequences x^t=x_1...x_t, i.e. P(St=i , X^t=x^t).
        Follows the method by Zucchini A.1.8 p 334.

        Returns
        -------
        log-likelihood: float
            log-likehood of given HMM parameters
        log of forward probabilities: ndarray of shape (n_samples, n_states)
            Array of the log of forward probabilties at each time step
        """
        n_obs, n_states = self.log_emission_probs_.shape
        log_alphas = np.zeros((n_obs, n_states))

        # Do the pass in cython
        hmm_cython.forward_proba(n_obs, n_states,
                      np.log(self.delta),
                      np.log(self.T),
                      self.log_emission_probs_, log_alphas)
        return logsumexp(log_alphas[-1]), log_alphas  # log-likelihood and forward probabilities

    def _log_backward_proba(self):
        """
        Compute the log of backward probabilities in scaled form.
        Backward probabilities are the conditional probability of
        some observation at t+1 given the current state = i. Equivalent to P(X_t+1 = x_t+1 | S_t = i)
        Returns
        -------
        log of backward probabilities: ndarray of shape (n_samples, n_states)
        """
        n_obs, n_states = self.log_emission_probs_.shape
        log_betas = np.zeros((n_obs, n_states))

        # Do the pass in cython
        hmm_cython.backward_proba(n_obs, n_states,
                                  np.log(self.delta),
                                  np.log(self.T),
                                  self.log_emission_probs_, log_betas)
        return log_betas

    def emission_probs(self, X):
        """
        Computes the probability distribution p(x) given an observation sequence X
        The calculation will return a T X N matrix

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Time series of data

        Returns
        ----------
        probs : ndarray of shape (n_samples, n_states)
            Output the probability for sampling from a particular state distribution  # TODO vend lige med CS ang. notation.
        """

        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init T X N matrix
        probs = np.zeros((T, self.n_states))

        # For all states evaluate the density function
        for j in range(self.n_states):
            log_probs[:, j] = stats.norm.logpdf(X, loc=self.mu[j], scale=self.std[j])

        probs = np.exp(log_probs)

        self.emission_probs_ = probs
        self.log_emission_probs_ = log_probs

        return probs, log_probs

    def _viterbi(self):
        n_obs, n_states = self.log_emission_probs_.shape
        log_alphas = np.zeros((n_obs, n_states))
        state_sequence = hmm_cython.viterbi(n_obs, n_states,
                                                        np.log(self.delta),
                                                        np.log(self.T),
                                                        self.log_emission_probs_)
        return state_sequence

    def sample(self, n_samples: int):
        '''
        Function samples states from a fitted Hidden Markov Model.

        Parameters
        ----------
        n_samples: ndarray of shape (n_samples,)
            Amount of samples to generate

        Returns
        -------
        samples : ndarray of shape (n_samples,)
            Outputs the generated samples of size n_samples
        '''

        state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=int)  # Array of possible states
        sample_states = np.zeros(n_samples).astype(int)  # Init sample vector
        sample_states[0] = np.random.choice(a=state_index, size=1, p=self.stationary_dist)  # First state is determined by initial dist

        for t in range(1, n_samples):
            # Each new state is chosen using the transition probs corresponding to the previous state sojourn.
            sample_states[t] = np.random.choice(a=state_index, size=1, p=self.T[sample_states[t - 1], :])

        samples = stats.norm.rvs(loc=self.mu[sample_states], scale=self.std[sample_states], size=n_samples)

        return samples, sample_states

    def decode(self):
        """
        Function to output the most likely sequence of states given an observation sequence.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Times series of data

        Returns
        ----------
        state_preds : ndarray of shape (n_samples,)
            Predicted sequence of states with length of the inputted time series.
        posteriors : ndarray of shape (n_samples, n_states)
            Computes the most likely state at each time-step, however, the state might not be valid (non-Viterbi) # TODO confirm with CS
        """
        state_preds = self._viterbi()
        return state_preds

    def predict_proba(self, X, n_preds=1):
        """
        Compute the probability P(St+h = i | X^T = x^T).
        Calculates the probability of being in state i at future time step h given a specific observation sequence up untill time T.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Time series of data
        n_preds : int, default=1
            Number of time steps to look forward from current time

        Returns
        ----------
        state_preds : ndarray of shape (n_states, n_preds)
            Output the probability of being in state i at time t+h
        """

        llk, log_alphas = self._log_forward_proba() # Compute scaled log-likelihood
        state_pred_t = np.exp(log_alphas[-1, :] - llk)

        state_preds = np.zeros(shape=(n_preds, self.n_states))  # Init matrix of predictions
        for t in range(n_preds):
            state_pred_t = state_pred_t @ self.T
            state_preds[t] = state_pred_t

        return state_preds

    def get_params_from_seq(self, X: ndarray, state_sequence: ndarray):  # TODO remove forward-looking params and slice X accordingly
        """
        Stores and outputs the model parameters

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Time series of data

        state_sequence : ndarray of shape (n_samples)
            State sequence for a given observation sequence
        """

        # Slice data
        if X.ndim == 1:  # Makes function compatible on higher dimensions
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

        # init dist and stationary dist
        self.delta = np.zeros(self.n_states)
        self.delta[state_sequence[0]] = 1.

        # Conditional distributions
        self.mu = state_groupby['X'].mean().values.T  # transform mean back into 1darray
        self.std = state_groupby['X'].std().values.T

    def get_stationary_dist(self):  # TODO not finished
        """
        Outputs the stationary distribution of the fitted model
        """
        ones = np.ones(shape=(self.n_states, self.n_states))
        identity = np.diag(ones)
        init_guess = np.ones(self.n_states) / self.n_states

        def solve_stationary(stationary_dist):
            return (stationary_dist @ (identity - self.T + ones)) - np.ones(self.n_states)

        stationary_dist = opt.root(solve_stationary, x0=init_guess)
        print(stationary_dist)

        self.stationary_dist = stationary_dist

    def _l2_norm_squared(self, z, theta):  # TODO this function is called too many times can we rewrite it somehow?
        # z must always be a vector but theta can be either a vector or a matrix
        # Subtract z from theta row-wise. Requires the transpose of the column matrix theta
        diff = (theta.T - z).T
        return np.square(np.linalg.norm(diff, axis=0))  # squared l2 norm.


if __name__ == '__main__':
    X = np.arange(1,1000)

    model = BaseHiddenMarkov(n_states=2)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions
    model._init_params(returns)
    model.emission_probs(returns)


    print(model.predict_proba(X,n_preds=1))

    model.stationary_dist = np.array([.5, .5])
    print(model.sample(10))

