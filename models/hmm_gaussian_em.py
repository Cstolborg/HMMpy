import numpy as np
from numpy import ndarray
from scipy.special import logsumexp
from scipy import stats
import matplotlib.pyplot as plt

from typing import List

from utils.simulate_returns import simulate_2state_gaussian
from models.hmm_base import BaseHiddenMarkov

import pyximport; pyximport.install()
from models.hmm_cython import forward, backward

''' TODO
FIT method does not choose best epoch
    - Implement same procedure as in hmm_jump.py


Move key algos into cython

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

    Methods
    -------
    fit


    Can be used to fit HMM parameters or to decode hidden states.

    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: int = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

    def _log_backward_proba(self, X: ndarray, emission_probs: ndarray):
        """ Compute the log of backward probabilities in scaled form.
        Backward probabilities are the conditional probability of
        some observation at t+1 given the current state = i. Equivalent to P(X_t+1 = x_t+1 | S_t = i)
        """
        T = len(X)
        log_betas = np.zeros((T, self.n_states))  # initialize matrix with zeros

        beta_t = np.ones(self.n_states) * 1 / self.n_states  # TODO CHECK WHY WE USE 1/M rather than ones
        llk = np.log(self.n_states)
        log_betas[-1, :] = np.log(np.ones(self.n_states))  # Last result is 0 since log(1)=0

        for t in range(T - 2, -1, -1):  # Count backwards
            beta_t = (self.T * emission_probs[t+1, :]) @ beta_t
            log_betas[t, :] = llk + np.log(beta_t)
            sum_beta_t = np.sum(beta_t)
            beta_t = beta_t / sum_beta_t  # Scale rows to sum to 1
            llk = llk + np.log(sum_beta_t)

        return log_betas

    def _do_forward_pass(self):
        n_samples, n_components = self.emission_probs_.shape
        work_buffer = np.zeros(self.n_states)
        fwdlattice = np.zeros((n_samples, n_components))
        forward(n_samples, n_components,
                            work_buffer,
                            self.delta,
                            self.T,
                            self.emission_probs_, fwdlattice)
        with np.errstate(under="ignore"):
            return fwdlattice

    def _do_backward_pass(self):
        n_samples, n_components = self.emission_probs_.shape
        work_buffer = np.zeros(self.n_states)
        bwdlattice = np.zeros((n_samples, n_components))
        forward(n_samples, n_components,
                            work_buffer,
                            self.delta,
                            self.T,
                            self.emission_probs_, bwdlattice)
        with np.errstate(under="ignore"):
            return bwdlattice

    def compute_gamma(self, log_alphas, log_betas):
        gamma = log_alphas + log_betas
        normalizer = logsumexp(gamma, axis=1, keepdims=True)
        gamma -= normalizer
        return np.exp(gamma)

    def _e_step(self, X: ndarray):
        ''' Do a single e-step in Baum-Welch algorithm

        '''
        T = len(X)
        self.emission_probs_, self.log_emission_probs_ = self.emission_probs(X)
        log_alphas = self._log_forward_proba(X, self.emission_probs_)
        log_betas = self._log_backward_proba(X, self.emission_probs_)

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

    def _m_step(self, X: ndarray, u, f, iterations: int = None):
        ''' Given u and f do an m-step.

         Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        X = np.array(X)

        # Update transition matrix and initial probs
        self.T = f / np.sum(f, axis=1).reshape((-1, 1))  # Check if this actually sums correct and to 1 on rows
        self.delta = u[0, :] / np.sum(u[0, :])

        # Update state-dependent distributions
        for j in range(self.n_states):
            self.mu[j] = np.sum(u[:, j] * X) / np.sum(u[:, j])
            self.std[j] = np.sqrt(np.sum(u[:, j] * np.square(X - self.mu[j])) / np.sum(u[:, j]))

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

        for epoch in range(self.epochs):
            # Do multiple random runs
            if epoch > 1: self._init_params(X, output_hmm_params=True)
            #print(f'Epoch {epoch} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')

            for iter in range(self.max_iter):
                # Do e- and m-step
                u, f, llk = self._e_step(X)
                self._m_step(X, u, f)

                # Check convergence criterion
                crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
                if crit < self.tol:
                    # Compute AIC and BIC and print model results
                    # AIC & BIC computed as shown on
                    # https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
                    num_independent_params = self.n_states**2 + 2*self.n_states - 1  # True for normal distributions
                    self.aic_ = -2 * llk + 2 * num_independent_params
                    self.bic_ = -2 * llk + num_independent_params * np.log(len(X))

                    if verbose == 1:
                        print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')
                    break

                elif iter == self.max_iter - 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_llk = llk


if __name__ == '__main__':
    model = EMHiddenMarkov(n_states=2, init="random", random_state=42)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions

    model.fit(returns, verbose=0)
    states, posteriors = model.decode(returns)

    log_alphas = model._do_forward_pass()
    log_betas = model._do_backward_pass()
    print(model.compute_gamma(log_alphas, log_betas))

    u,f,llk = model._e_step(returns)
    print(u)



    #print(model._log_forward_proba(returns, model.emission_probs_))



    '''
    #model.fit(returns, verbose=0)
    #states, posteriors = model._viterbi(returns)

    n_states = model.n_states
    emission_probs = model.emission_probs(returns)
    delta = model.delta
    TPM = model.T


    #print(_log_forward_probs(n_states, returns, emission_probs, delta, TPM) )
    '''


    plotting = False
    if plotting == True:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
        ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
        ax[1].plot(states, label='Predicted states', ls='dotted')
        ax[1].plot(true_regimes, label='True states', ls='dashed')

        plt.legend()
        plt.show()






    check_hmmlearn = False
    if check_hmmlearn == True:
        from hmmlearn import hmm

        hmmlearn = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(returns.reshape(-1, 1))

        print(hmmlearn.transmat_)
        print(hmmlearn.means_)
        print(hmmlearn.covars_)

        predictions = hmmlearn.predict(returns.reshape(-1, 1))
