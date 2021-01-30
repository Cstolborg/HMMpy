import numpy as np
from numpy import ndarray
from scipy.special import logsumexp
from scipy import stats
import matplotlib.pyplot as plt

from typing import List

from utils.simulate_returns import simulate_2state_gaussian, plot_posteriors_states
from models.hmm_base import BaseHiddenMarkov

import pyximport; pyximport.install()  # TODO can only be active during development -- must be done through setup.py
from models import hmm_cython

''' TODO
Write code clean
    - Many methods take X as input but only use it to get length

'''

class EMHiddenMarkov(BaseHiddenMarkov):
    """ Class for computing HMM's using the EM algorithm.

    Parameters
    ----------
    n_states : int, default=2
            Number of hidden states
    max_iter : Maximum number of iterations to perform during expectation-maximization
    tol : Criterion for early stopping
    init: str
            Set to 'random' for random initialization.
            Set to None for deterministic init.

   Attributes
    ----------
    Can be used to fit HMM parameters or to decode hidden states.

    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: float = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

    def _log_forward_proba(self):
        n_obs, n_states = self.log_emission_probs_.shape
        log_alphas = np.zeros((n_obs, n_states))

        # Do the pass in cython
        hmm_cython.forward_proba(n_obs, n_states,
                      np.log(self.delta),
                      np.log(self.T),
                      self.log_emission_probs_, log_alphas)
        return logsumexp(log_alphas[-1]), log_alphas  # log-likelihood and forward probabilities

    def _log_backward_proba(self):
        n_obs, n_states = self.log_emission_probs_.shape
        log_betas = np.zeros((n_obs, n_states))

        # Do the pass in cython
        hmm_cython.backward_proba(n_obs, n_states,
                                  np.log(self.delta),
                                  np.log(self.T),
                                  self.log_emission_probs_, log_betas)
        return log_betas

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
        ndarray
        """
        gamma = log_alphas + log_betas
        normalizer = logsumexp(gamma, axis=1, keepdims=True)
        gamma -= normalizer
        return np.exp(gamma)

    def _e_step(self, X: ndarray):
        '''
        Do a single e-step in Baum-Welch algorithm
        '''
        T = len(X)
        self.emission_probs_, self.log_emission_probs_ = self.emission_probs(X)
        llk, log_alphas = self._log_forward_proba()
        log_betas = self._log_backward_proba()

        gamma = self.compute_posteriors(log_alphas, log_betas)

        # Initialize matrix of shape j X j
        # Number of expected transitions from state i to j
        xi = np.zeros(shape=(self.n_states, self.n_states))
        for j in range(self.n_states):
            for k in range(self.n_states):
                xi[j, k] = self.T[j, k] * np.sum(
                    np.exp(log_alphas[:-1, j] + log_betas[1:, k] + self.log_emission_probs_[1:, k] - llk))

        return gamma, xi, llk

    def _m_step(self, X: ndarray, gamma, xi):
        ''' Given u and f do an m-step.

         Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        X = np.array(X)

        # Update transition matrix and initial probs
        self.T = xi / np.sum(xi, axis=1).reshape((-1, 1))  # Check if this actually sums correct and to 1 on rows
        self.delta = gamma[0, :] / np.sum(gamma[0, :])

        # Update state-dependent distributions
        for j in range(self.n_states):
            self.mu[j] = np.sum(gamma[:, j] * X) / np.sum(gamma[:, j])
            self.std[j] = np.sqrt(np.sum(gamma[:, j] * np.square(X - self.mu[j])) / np.sum(gamma[:, j]))

    def fit(self, X: ndarray, verbose=0):
        """
        Iterates through the e-step and the m-step.
        Parameters
        ----------
        X
        verbose

        Returns
        -------

        """
        # Init parameters initial distribution, transition matrix and state-dependent distributions
        self._init_params(X, output_hmm_params=True)
        self.old_llk = -np.inf  # Used to check model convergence
        self.best_epoch = -np.inf

        for epoch in range(self.epochs):
            # Do new init at each epoch
            if epoch > 0: self._init_params(X, output_hmm_params=True)

            for iter in range(self.max_iter):
                # Do e- and m-step
                gamma, xi, llk = self._e_step(X)
                self._m_step(X, gamma, xi)

                # Check convergence criterion
                crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
                if crit < self.tol:
                    if llk > self.best_epoch:
                        # Compute AIC and BIC and print model results
                        # AIC & BIC computed as shown on
                        # https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
                        num_independent_params = self.n_states ** 2 + 2 * self.n_states - 1  # True for normal distributions
                        self.aic_ = -2 * llk + 2 * num_independent_params
                        self.bic_ = -2 * llk + num_independent_params * np.log(len(X))
                        self.best_T = self.T
                        self.best_delta = self.delta
                        self.best_mu = self.mu
                        self.best_std = self.std

                    if verbose == 1:
                        print(
                            f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')
                    break

                elif iter == self.max_iter - 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_llk = llk

        self.T = self.best_T
        self.delta = self.best_delta
        self.mu = self.best_mu
        self.std = self.best_std

    def _e_step1(self, X: ndarray):
        ''' Do a single e-step in Baum-Welch algorithm

        '''
        T = len(X)
        self.emission_probs_, self.log_emission_probs_ = self.emission_probs(X)
        log_alphas = self._log_forward_probs(X, self.emission_probs_)
        log_betas = self._log_backward_probs(X, self.emission_probs_)

        # Compute scaled log-likelihood
        llk_scale_factor = np.max(log_alphas[-1, :])  # Max of the last vector in the matrix log_alpha
        llk = llk_scale_factor + np.log(
            np.sum(np.exp(log_alphas[-1, :] - llk_scale_factor)))  # Scale log-likelihood by c

        # Expectation of being in state j given a sequence x^T
        # P(S_t = j | x^T)
        u = np.exp(log_alphas + log_betas - llk)  # TODO FIND BETTER VARIABLE NAME

        # Initialize matrix of shape j X j
        # We skip computing vhat and head straight to fhat for computational reasons
        f = np.zeros(shape=(self.n_states, self.n_states))  # TODO FIND BETTER VARIABLE NAME
        for j in range(self.n_states):
            for k in range(self.n_states):
                f[j, k] = self.T[j, k] * np.sum(
                    np.exp(log_alphas[:-1, j] + log_betas[1:, k] + self.log_emission_probs_[1:, k] - llk))

        return u, f, llk

    def fit1(self, X: ndarray, verbose=0):
        """
        Iterates through the e-step and the m-step.
        Parameters
        ----------
        X
        verbose

        Returns
        -------

        """
        # Init parameters initial distribution, transition matrix and state-dependent distributions
        self._init_params(X, output_hmm_params=True)
        self.old_llk = -np.inf  # Used to check model convergence
        self.best_epoch = -np.inf

        for epoch in range(self.epochs):
            # Do multiple random runs
            if epoch > 1: self._init_params(X, output_hmm_params=True)
            #print(f'Epoch {epoch} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')

            for iter in range(self.max_iter):
                # Do e- and m-step
                u, f, llk = self._e_step1(X)
                self._m_step(X, u, f)

                # Check convergence criterion
                crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
                if crit < self.tol:
                    if llk > self.best_epoch:
                        # Compute AIC and BIC and print model results
                        # AIC & BIC computed as shown on
                        # https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
                        num_independent_params = self.n_states**2 + 2*self.n_states - 1  # True for normal distributions
                        self.aic_ = -2 * llk + 2 * num_independent_params
                        self.bic_ = -2 * llk + num_independent_params * np.log(len(X))
                        self.best_T = self.T
                        self.best_delta = self.delta
                        self.best_mu = self.mu
                        self.best_std = self.std


                    if verbose == 1:
                        print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')
                    break

                elif iter == self.max_iter - 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_llk = llk
        self.T = self.best_T
        self.delta = self.best_delta
        self.mu = self.best_mu
        self.std = self.best_std

        return iter


if __name__ == '__main__':
    model = EMHiddenMarkov(n_states=2, init="random", random_state=1, epochs=1, max_iter=100)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions

    model.fit(returns, verbose=1)

    states = model.decode()
    #states, posteriors = model.decode(returns)

    llk, log_alphas = model._log_forward_proba()
    log_betas = model._log_backward_proba()

    posteriors = model.compute_posteriors(log_alphas, log_betas)

    print(posteriors)
    plot_posteriors_states(posteriors, states, true_regimes)


    check_hmmlearn = False
    if check_hmmlearn == True:
        from hmmlearn import hmm

        hmmlearn = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(returns.reshape(-1, 1))

        print(hmmlearn.transmat_)
        print(hmmlearn.means_)
        print(hmmlearn.covars_)

        predictions = hmmlearn.predict(returns.reshape(-1, 1))
