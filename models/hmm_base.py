import numpy as np
from numpy.core._multiarray_umath import ndarray
from scipy import stats
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.cluster._kmeans import kmeans_plusplus
import matplotlib.pyplot as plt


from utils.simulate_returns import simulate_2state_gaussian

import pyximport; pyximport.install()  # TODO can only be active during development -- must be done through setup.py
from models import hmm_cython

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
            Set to 'kmeans++' to use that init method - only supported for jump models.
            Set to 'random' for random initialization.
            Set to "deterministic" for deterministic init.
    Attributes
    ----------
    mu : ndarray of shape (n_states,)
        Fitted means for each state
    std : ndarray of shape (n_states,)
        Fitted std for each state
    tpm : ndarray of shape (n_states, n_states)
        Transition probability matrix between states
    start_proba : ndarray of shape (n_states,)
        Initial state occupation distribution
    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: float = 1e-6,
                 epochs: int = 1, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.max_iter = max_iter  # Max iterations to fit model
        self.epochs = epochs  # Set number of random inits used in model fitting
        self.tol = tol
        self.init = init

        self.stationary_dist = None
        self.tpm = None
        self.start_proba = None
        self.mu = None
        self.std = None

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

            self.tpm = trans_prob
            self.start_proba = init_dist
            self.mu = mu
            self.std = std

        elif self.init == "deterministic":
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

            self.tpm = trans_prob
            self.start_proba = init_dist
            self.mu = mu
            self.std = std

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

    def sample(self, n_samples, n_sequences=1, hmm_params=None):
        '''
        Sample states from a fitted Hidden Markov Model.

        Parameters
        ----------
        n_samples: int
            Amount of samples to generate
        hmm_params: dict, default=None
            hmm model parameters to sample from. If None and model is fitted it will use fitted parameters.
            To manually set params, create a dict with 'mu', 'std', 'tpm' and 'stationary distribution' as kwds
            and values ndarrays.

        Returns
        -------
        samples : ndarray of shape (n_samples,)
            Outputs the generated samples of size n_samples
        '''
        if hmm_params == None:
            mu = self.mu
            std = self.std
            tpm = self.tpm
            stationary_dist = self.stationary_dist
        else:
            mu = hmm_params['mu']
            std = hmm_params['std']
            tpm = hmm_params['tpm']
            stationary_dist = hmm_params['stationary_dist']


        state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=np.int32)  # Array of possible states
        sample_states = np.zeros(shape=(n_samples, n_sequences), dtype=np.int32) # Init sample vector
        samples = np.zeros(shape=(n_samples, n_sequences))  # Init sample vector

        for seq in range(n_sequences):
            sample_states[0, seq] = np.random.choice(a=state_index, size=1, p=stationary_dist)

            for t in range(1, n_samples):
                # Each new state is chosen using the transition probs corresponding to the previous state sojourn.
                sample_states[t, seq] = np.random.choice(a=state_index, size=1, p=tpm[sample_states[t - 1, seq], :])

            samples[:, seq] = stats.norm.rvs(loc=mu[sample_states[:, seq]], scale=std[sample_states[:, seq]], size=n_samples)

        if n_sequences == 1:
            sample_states = sample_states[:, 0]
            samples = samples[:, 0]

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

    def predict_proba(self, n_preds=1):
        """
        Compute the probability P(St+h = i | X^T = x^T).
        Calculates the probability of being in state i at future time step h given a specific observation sequence up untill time T.

        Parameters
        ----------
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
            state_pred_t = state_pred_t @ self.tpm
            state_preds[t] = state_pred_t

        return state_preds

    def get_stationary_dist(self, tpm):
        """
        Outputs the stationary distribution of the fitted model.

        The stationary distributions corresponds to the largest eigenvector of the
        transition probability matrix. Since all values in the TPM are bounded between 0 and 1,
        we know that the largest eigenvalue is 1, and that the eigenvectors will all be defined by real numbers.

        Computed by taking the eigenvector corresponding to the largest eigenvalue and scaling it to sum to 1.

        Returns
        -------
        stationary_dist: ndarray of shape (n_states,)
        """
        # Computes right eigenvectors, thus transposing the TPM is necessary.
        eigvals, eigvecs = np.linalg.eig(tpm.T)
        eigvec = eigvecs[:, np.argmax(eigvals)]  # Get the eigenvector corresponding to largest eigenvalue

        stationary_dist = eigvec / eigvec.sum()
        return stationary_dist

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
                                 np.log(self.start_proba),
                                 np.log(self.tpm),
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
                                  np.log(self.start_proba),
                                  np.log(self.tpm),
                                  self.log_emission_probs_, log_betas)
        return log_betas

    def _viterbi(self):
        n_obs, n_states = self.log_emission_probs_.shape
        log_alphas = np.zeros((n_obs, n_states))
        state_sequence = hmm_cython.viterbi(n_obs, n_states,
                                            np.log(self.start_proba),
                                            np.log(self.tpm),
                                            self.log_emission_probs_)
        return state_sequence


if __name__ == '__main__':
    X = np.arange(1,1000)

    model = BaseHiddenMarkov(n_states=2)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions
    model._init_params(returns)
    model.emission_probs(returns)


