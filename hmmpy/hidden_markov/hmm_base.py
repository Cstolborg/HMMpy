import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from hmmpy.hidden_markov import hmm_cython

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
        Set to 'kmeans++' to use that init method - only supported for jump hidden_markov.
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
        self.is_fitted = False

        np.random.seed(self.random_state)

    def _init_params(self, X=None, diag_uniform_dist = (.95, .99), output_hmm_params=True):
        """
        Function to initialize HMM parameters. Can do so using kmeans++, randomly or deterministic.

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
            np.fill_diagonal(trans_prob, trans_prob.diagonal() - remaining_row_nums)  # And subtract these numbers from the diagonal so it remains uniform

            # initial distribution
            init_dist = np.ones(self.n_states) / np.sum(self.n_states)  # initial distribution 1 X N vector

            # State dependent distributions
            mu = np.random.uniform(low=-0.05, high=0.15, size=self.n_states)  # np.random.rand(self.n_states)
            std = np.random.uniform(low=0.01, high=0.3, size=self.n_states)  # np.random.rand(self.n_states) #

            self.tpm = trans_prob
            self.start_proba = init_dist
            self.mu = mu
            self.std = std

        elif self.init == "deterministic":  # TODO deprecate
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
            Probability for sampling from a particular state distribution
        log_probs : ndarray of shape (n_samples, n_states)
            Log probability for sampling from a particular state distribution
        """

        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init T X N matrix
        probs = np.zeros((T, self.n_states))

        # For all states evaluate the density function
        for j in range(self.n_states):
            if self.std[j] == np.nan or self.std[j] == 0. or self.std[j] < 0.:
                continue  # If std has non-standard value keep the probs at zero and go to next loop
            log_probs[:, j] = stats.norm.logpdf(X, loc=self.mu[j], scale=self.std[j])

        probs = np.exp(log_probs)

        self.emission_probs_ = probs
        self.log_emission_probs_ = log_probs

        return probs, log_probs

    def fit_predict(self, X, n_preds=15, verbose=False):
        """
        Fit model, then decode states and make n predictions.
        Wraps .fit(), .decode() and .predict_proba() into one method.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Data to be fitted.
        n_preds : int, default=15
            Number of time steps to look forward from current time

        Returns
        -------

        """
        self.fit(X, sort_state_seq=True, verbose=False,)
        if self.is_fitted == False:  # Check if model is fitted
            max_iter = self.max_iter
            self.max_iter = max_iter * 2  # Double amount of iterations
            self.fit(X)  # Try fitting again
            self.max_iter = max_iter  # Reset max_iter back to user-input
            if self.is_fitted == False and verbose == True:
                print(f'NOT FITTED -- mu {self.mu} -- std {self.std} -- tpm {np.diag(self.tpm)}')

        if self.type == 'mle':
            state_sequence = self.decode(X)  # 1D-array with most likely state sequence
        elif self.type == 'jump':
            state_sequence = self.state_seq

        # Posterior probability of being in state j at time t
        self.emission_probs_, self.log_emission_probs_ = self.emission_probs(X)
        posteriors = self.predict_proba(n_preds)  # 2-D array of shape (n_preds, n_states)

        return state_sequence, posteriors

    def sample(self, n_samples, n_sequences=1, hmm_params=None):
        '''
        Sample states from a fitted Hidden Markov Model.

        Parameters
        ----------
        n_samples : int
            Amount of samples to generate
        n_sequences : int, default=1
            Number of independent sequences to sample from, e.g. if n_samples=100 and n_sequences=3
            then 3 different sequences of length 100 are sampled
        hmm_params : dict, default=None
            hmm model parameters to sample from. If None and model is fitted it will use fitted parameters.
            To manually set params, create a dict with 'mu', 'std', 'tpm' and 'stationary distribution' as kwds
            and values ndarrays.

        Returns
        -------
        samples : ndarray of shape (n_samples,)
            Outputs the generated samples of size n_samples
        sample_states : ndarray of shape (n_samples, n_sequences)
            Outputs sampled states
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

            try: # Prevents program from crashing due to "Domain errors"
                samples[:, seq] = stats.norm.rvs(loc=mu[sample_states[:, seq]], scale=std[sample_states[:, seq]], size=n_samples)
            except Exception:
                print(sys.exc_info()[0])
                print(f'Error occurrred with params: MU {self.mu} -- STD {self.std}, -- TPM {self.tpm.ravel()}')

        if n_sequences == 1:
            sample_states = sample_states[:, 0]
            samples = samples[:, 0]

        return samples, sample_states

    def decode(self, X):
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
        """
        state_preds = self._viterbi(X)
        return state_preds

    def predict_proba(self, n_preds=15):
        """
        Compute the probability P(St+h = i | X^T = x^T).
        Calculates the probability of being in state i at future time step h given a specific observation sequence up untill time T.

        Parameters
        ----------
        n_preds : int, default=15
            Number of time steps to look forward from current time

        Returns
        ----------
        state_preds : ndarray of shape (n_states, n_preds)
            Output the probability of being in state i at time t+h
        """

        llk, log_alphas = self._log_forward_proba() # Compute scaled log-likelihood

        # Same as posterior at terminal time. Subtracting llk to get unscaled probability.
        # Since backward_proba = 1 at last time step, this is omitted
        state_pred_t = np.exp(log_alphas[-1, :] - llk)

        state_preds = np.zeros(shape=(n_preds, self.n_states))  # Init matrix of predictions
        for t in range(n_preds):
            state_preds[t] = state_pred_t = state_pred_t @ self.tpm

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
        if tpm.ndim < 2:  # In cases where only 1 state is detected
            return np.array([1., 0.])

        # Computes right eigenvectors, thus transposing the TPM is necessary.
        eigvals, eigvecs = np.linalg.eig(tpm.T)
        eigvec = eigvecs[:, np.argmax(eigvals)]  # Get the eigenvector corresponding to largest eigenvalue

        stationary_dist = eigvec / eigvec.sum()
        return stationary_dist

    def _log_forward_proba(self):
        r"""
        Compute log forward probabilities in scaled form.

        Forward probability is essentially the joint probability of observing
        a state = i and observation sequences x^t=x_1...x_t, i.e. $P(S_t=i , X^t=x^t)$.
        Follows the method by Zucchini A.1.8 p 334.

        Returns
        -------
        log-likelihood : float
            log-likehood of given HMM parameters
        log of forward probabilities : ndarray of shape (n_samples, n_states)
            Array of the scaled log of forward probabilities at each time step.
        """
        n_obs, n_states = self.log_emission_probs_.shape
        log_alphas = np.zeros((n_obs, n_states))

        self.check_params_not_zero()

        # Do the pass in cython
        with np.errstate(divide='ignore'):
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
        with np.errstate(divide='ignore'):
            hmm_cython.backward_proba(n_obs, n_states,
                                      np.log(self.start_proba),
                                      np.log(self.tpm),
                                      self.log_emission_probs_, log_betas)
            return log_betas

    def _viterbi(self, X):
        self.emission_probs_, self.log_emission_probs_ = self.emission_probs(X)
        n_obs, n_states = self.log_emission_probs_.shape
        state_sequence = hmm_cython.viterbi(n_obs, n_states,
                                            np.log(self.start_proba),
                                            np.log(self.tpm),
                                            self.log_emission_probs_)
        return state_sequence.astype(np.int32)

    def check_params_not_zero(self):
        if self.start_proba[0] == 0:
            self.start_proba[0] = 0.0000001
            self.start_proba[1] = 0.9999999
        elif self.start_proba[1] == 0:
            self.start_proba[1] = 0.0000001
            self.start_proba[0] = 0.9999999

        if self.tpm[0, 0] == 1.:
            self.tpm[0, 0] = 0.999999
            self.tpm[0, 1] = 0.000001
        elif self.tpm[1, 1] == 1.:
            self.tpm[1, 1] = 0.999999
            self.tpm[1, 0] = 0.000001

    def fit(self, X, get_hmm_params=True, sort_state_seq=True, verbose=False, feature_set='feature_set_2'):
        """
        fit model to data. Defined in respective child classes
        """
        pass

    def rolling_posteriors(self, X):
        """ Compute the posterior probability of being in state i at time T.

        Function should be used as part of rolling estimation when one is
        interested only in the final smoothing probability of the current window sample.
        """
        if not self.is_fitted is True:
            print('Warning. Trying to predict using an unfitted model.')

        # Use emission probs to do a forward pass
        # Note only forward probs are needed as backward probs at time T is 1 by definition.
        self.emission_probs(X)
        llk, log_alphas = self._log_forward_proba()

        # Probability of being in state i in time T
        posterior = np.exp(log_alphas[-1] - llk)
        return posterior

    def bac_score(self, X, y_true, verbose=False):
        """ Computes balanced accuracy score when true state sequence is known """
        if not self.is_fitted is True:
            bac = 0.
            return bac
        if self.type == 'mle' or self.type == 'sampler':
            y_pred = self.decode(X)
        elif self.type == 'jump':
            y_pred = self.state_seq
            y_true = y_true[(self.window_len[-1] - 1):]  # slice y_true to have same dim as y_pred

        conf_matrix = confusion_matrix(y_true, y_pred)
        keep_idx = conf_matrix.sum(axis=1) != 0  # Only keep non-zero rows
        conf_matrix = conf_matrix[keep_idx]

        tp = np.diag(conf_matrix)
        fn = conf_matrix.sum(axis=1) - tp
        tpr = tp / (tp + fn)
        bac = np.mean(tpr)

        if verbose is True:
            logical_1 = bac < 0.5
            logical_2 = conf_matrix.ndim > 1 and \
                        np.any(conf_matrix.sum(axis=1)==0)

            if logical_1 or logical_2:
                print(f'bac {bac} -- tpr {tpr}')
                print(conf_matrix)

        return bac


if __name__ == '__main__':
    # Including data to train on S&P 500
    path_1 = '../../data/'
    df_returns = pd.read_csv(path_1 + 'price_series.csv', index_col='Time')
    df_returns.index = pd.to_datetime(df_returns.index)
    df_SP500 = df_returns[['S&P 500 ']]
    df_SP500['S&P 500 Index'] = df_SP500['S&P 500 '] / df_SP500['S&P 500 '][0] * 100
    df_SP500['Returns'] = df_SP500['S&P 500 Index'].pct_change()
    df_SP500['Log returns'] = np.log(df_SP500['S&P 500 Index']) - np.log(df_SP500['S&P 500 Index'].shift(1))
    df_SP500.dropna(inplace = True)
    print(df_SP500)

    #X = np.arange(1,1000)

    X = df_SP500['S&P 500 ']
    model = BaseHiddenMarkov(n_states=2)
    #returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions
    model._init_params(df_SP500['Log returns'])
    probs, logprobs = model.emission_probs(df_SP500['Log returns'])
    print(probs)

    model.fit(X)
