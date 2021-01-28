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
    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: int = 1e-6,
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

        Returns
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

    def decode(self, X):
        state_preds, posteriors = self._viterbi(X)
        return state_preds, posteriors

    def emission_probs(self, X):
        """ Compute all different probabilities p(x) given an observation sequence and n states

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

    def _log_forward_proba(self, X: ndarray, emission_probs: ndarray):
        """ Compute log forward probabilities in scaled form.

        Forward probability is essentially the joint probability of observing
        a state = i and observation sequences x^t=x_1...x_t, i.e. P(St=i , X^t=x^t).
        Follows the method by Zucchini A.1.8 p 334.
        """
        T = len(X)
        log_alphas = np.zeros((T, self.n_states))  # initialize matrix with zeros

        # a0, compute first forward as dot product of initial dist and state-dependent dist
        # Each element is scaled to sum to 1 in order to handle numerical underflow
        alpha_t = self.delta * emission_probs[0, :]
        sum_alpha_t = np.sum(alpha_t)
        alpha_t_scaled = alpha_t / sum_alpha_t
        llk = np.log(sum_alpha_t)  # Scalar to store the log likelihood
        log_alphas[0, :] = llk + np.log(alpha_t_scaled)

        # a1 to at, compute recursively
        for t in range(1, T):
            alpha_t = (alpha_t_scaled @ self.T) * emission_probs[t, :]  # Dot product of previous forward_prob, transition matrix and emmission probablitites
            sum_alpha_t = np.sum(alpha_t)

            alpha_t_scaled = alpha_t / sum_alpha_t  # Scale forward_probs to sum to 1
            llk = llk + np.log(sum_alpha_t)  # Scalar to store likelihoods
            log_alphas[t, :] = llk + np.log(alpha_t_scaled)

        return log_alphas

    def predict_proba(self, X, n_preds=1):
        """
        Compute the probability P(St+h = i | X^T = x^T).
        Calculates the probability of being in state i at future time step h.

        Parameters
        ----------
        X
        n_preds

        Returns
        -------
        """

        log_alphas = self._log_forward_proba(X, self.emission_probs_)
        # Compute scaled log-likelihood
        llk_scale_factor = np.max(log_alphas[-1, :])  # Max of the last vector in the matrix log_alpha
        llk = llk_scale_factor + np.log(
            np.sum(np.exp(log_alphas[-1, :] - llk_scale_factor)))  # Scale log-likelihood by c
        state_pred_t = np.exp(log_alphas[-1, :] - llk)

        state_preds = np.zeros(shape=(n_preds, self.n_states))  # Init matrix of predictions
        for t in range(n_preds):
            state_pred_t = state_pred_t @ self.T
            state_preds[t] = state_pred_t

        return state_preds

    def get_params_from_seq(self, X: ndarray, state_sequence: ndarray):  # TODO remove forward-looking params and slice X accordingly
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
    model = BaseHiddenMarkov(n_states=2)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions
    model._init_params(returns)

    print(model.mu)
    print(model.std)
    print(model.T)
    print(model.delta)

    model.stationary_dist = np.array([.5, .5])
    print(model.sample(10))
