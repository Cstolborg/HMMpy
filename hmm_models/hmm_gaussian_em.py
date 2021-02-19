import os
import numpy as np
from numpy import ndarray
from scipy.special import logsumexp
from scipy import stats
import matplotlib.pyplot as plt

from utils.simulate_returns import simulate_2state_gaussian
from hmm_models.hmm_base import BaseHiddenMarkov

from multiprocessing import Pool


class EMHiddenMarkov(BaseHiddenMarkov):
    """"
    Class for computing HMM's using the EM algorithm.
    Can be used to fit HMM parameters or to decode hidden states.

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
    init: str
        Set to 'random' for random initialization.
        Set to None for deterministic init.

    Attributes
    ----------
    mu : ndarray of shape (n_states,)
        Fitted means for each state
    std : ndarray of shape (n_states,)
        Fitted std for each state
    tpm : ndarray of shape (n_states, n_states)
        Matrix of transition probabilities between states
    start_proba : ndarray of shape (n_states,)
        Initial state occupation distribution
    gamma : ndarray of shape (n_states,)
        Entails the probability of being in a state at time t knowing
        all the observations that has come and all the observations to come. (Its a bowtie)
    aic_ : float
        Measurement to select the best fitted model
    bic_ : float
        Measurement to select the best fitted model
    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: float = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

    def compute_posteriors(self, log_alphas, log_betas):
        """
        Expectation of being in state j at time t given observations, P(S_t = j | x^T).
        Note to self: Same as gamma in Rabiners notation.

        Parameters
        ----------
        log_alphas
        log_betas

        Returns
        -------
        gamma : ndarray of shape (n_samples, n_states)
            Expectation of being in state j at time t given observations, P(S_t = j | x^T)
        """
        gamma = log_alphas + log_betas
        normalizer = logsumexp(gamma, axis=1, keepdims=True)
        gamma -= normalizer
        return np.exp(gamma)

    def _e_step(self, X: ndarray):
        '''
        Do a single e-step in Baum-Welch algorithm (Derives Xi and Gamma w.r.t. traditional HMM syntax)
        '''
        self.emission_probs(X)
        llk, log_alphas = self._log_forward_proba()
        log_betas = self._log_backward_proba()

        gamma = self.compute_posteriors(log_alphas, log_betas)

        # Initialize matrix of shape j X j
        # Number of expected transitions from state i to j
        xi = np.zeros(shape=(self.n_states, self.n_states))
        for j in range(self.n_states):
            for k in range(self.n_states):
                xi[j, k] = self.tpm[j, k] * np.sum(
                    np.exp(log_alphas[:-1, j] + log_betas[1:, k] + self.log_emission_probs_[1:, k] - llk))

        return gamma, xi, llk

    def _m_step(self, X: ndarray, gamma, xi):
        ''' 
        Given u and f do an m-step.
        Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        # Update transition matrix and initial probs
        self.tpm = xi / np.sum(xi, axis=1).reshape((-1, 1))  # Check if this actually sums correct and to 1 on rows
        self.start_proba = gamma[0, :] / np.sum(gamma[0, :])

        # Update state-dependent distributions
        for j in range(self.n_states):
            self.mu[j] = np.sum(gamma[:, j] * X) / np.sum(gamma[:, j])
            self.std[j] = np.sqrt(np.sum(gamma[:, j] * np.square(X - self.mu[j])) / np.sum(gamma[:, j]))

    def fit(self, X: ndarray, verbose=0):
        """
        Function iterates through the e-step and the m-step recursively to find the optimal model parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Time series of data
        verbose : boolean
            False / True for extra information regarding the function.

        Returns
        ----------
        Derives the optimal model parameters
        """
        # Init parameters initial distribution, transition matrix and state-dependent distributions
        self.is_fitted = False
        self._init_params(X, output_hmm_params=True)
        self.old_llk = -np.inf  # Used to check model convergence
        self.best_epoch = -np.inf

        for epoch in range(self.epochs):
            # Do new init at each epoch
            if epoch > 0:
                self._init_params(X, output_hmm_params=True)

            for iter in range(self.max_iter):
                # Do e- and m-step
                gamma, xi, llk = self._e_step(X)
                self._m_step(X, gamma, xi)

                # Check convergence criterion
                crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
                if crit < self.tol:
                    self.is_fitted = True
                    if llk > self.best_epoch:
                        # Compute AIC and BIC and print model results
                        # AIC & BIC computed as shown on
                        # https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
                        num_independent_params = self.n_states ** 2 + 2 * self.n_states - 1  # True for normal distributions
                        self.aic_ = -2 * llk + 2 * num_independent_params
                        self.bic_ = -2 * llk + num_independent_params * np.log(len(X))
                        self.stationary_dist = self.get_stationary_dist(tpm=self.tpm)

                        self.best_tpm = self.tpm
                        self.best_delta = self.start_proba
                        self.best_mu = self.mu
                        self.best_std = self.std

                    if verbose == 1:
                        print(
                            f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.tpm)} - Delta {self.start_proba}')
                    break

                elif iter == self.max_iter - 1 and verbose == 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_llk = llk

        self.tpm = self.best_tpm
        self.start_proba = self.best_delta
        self.mu = self.best_mu
        self.std = self.best_std



if __name__ == '__main__':
    model = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=1, max_iter=100)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions

    pool = Pool()

    result = pool.map(model.fit, [returns]*10)

    #plot_samples_states(sample_rets, sample_states)






    check_hmmlearn = False
    if check_hmmlearn == True:
        from hmmlearn import hmm

        hmmlearn = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(returns.reshape(-1, 1))

        print(hmmlearn.transmat_)
        print(hmmlearn.means_)
        print(hmmlearn.covars_)

        predictions = hmmlearn.predict(returns.reshape(-1, 1))
