import numpy as np
from numpy import ndarray
from scipy import stats
import matplotlib.pyplot as plt

from typing import List

from utils.simulate_returns import simulate_2state_gaussian
#from models.hmm_cython import _log_forward_probs
from models.hmm_base import BaseHiddenMarkov

''' TODO
FIT method does not choose best epoch
    - Implement same procedure as in hmm_jump.py


Move key algos into cython

'''

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
    T : ndarray of shape (n_states, n_states)
        Matrix of transition probabilities between states
    delta : ndarray of shape (n_states,)
        Initial state occupation distribution
    gamma : ndarray of shape (n_states,)
        Entails the probability of being in a state at time t knowing all the observations that has come and all the observations to come. (Its a bowtie)
    AIC : float
        Measurement to select the best fitted model
    BIC : float
        Measurement to select the best fitted model
    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: int = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

    def _log_backward_probs(self, X: ndarray, emission_probs: ndarray):
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

    def _e_step(self, X: ndarray):
        ''' Do a single e-step in Baum-Welch algorithm (Derives Xi and Gamma w.r.t. traditional HMM syntax)

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
        '''
        Function iterates through the e-step and the m-step recursively to find the optimal model parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Time series of data
        Verbose : boolean
            False / True for extra information regarding the function.

        Returns
        ----------
        Derives the optimal model parameters
        '''

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

                    print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')
                    break

                elif iter == self.max_iter - 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_llk = llk

                if verbose == 1: print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')


if __name__ == '__main__':
    model = EMHiddenMarkov(n_states=2, init="random", random_state=42)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions

    model.fit(returns)
    states, posteriors = model.decode(returns)

    print(model.predict_proba(returns, 5))



    model.fit(returns, verbose=0)
    states, posteriors = model._viterbi(returns)

    n_states = model.n_states
    emission_probs = model.emission_probs(returns)
    delta = model.delta
    TPM = model.T
    AIC = model.aic_
    BIC = model.bic_



    #print(n_states, returns, emission_probs, delta, TPM)
    print('BIC =', BIC)
    print('AIC = ', AIC)

    """"
    plotting = False
    if plotting == True:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
        ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
        ax[1].plot(states, label='Predicted states', ls='dotted')
        ax[1].plot(true_regimes, label='True states', ls='dashed')

        plt.legend()
        plt.show()
    """





    check_hmmlearn = False
    if check_hmmlearn == True:
        from hmmlearn import hmm

        hmmlearn = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(returns.reshape(-1, 1))

        print(hmmlearn.transmat_)
        print(hmmlearn.means_)
        print(hmmlearn.covars_)

        predictions = hmmlearn.predict(returns.reshape(-1, 1))
