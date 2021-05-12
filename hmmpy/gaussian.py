import numpy as np
from numpy import ndarray
from scipy.special import logsumexp

from hmmpy.base import BaseHiddenMarkov
from hmmpy.sampler import SampleHMM


class GaussianHMM(BaseHiddenMarkov):
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
        all the observations that has come and all the observations to come.
    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: float = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)
        self.type = 'mle'

    def compute_log_posteriors(self, log_alphas, log_betas):
        """
        Expectation of being in state j at time t given observations, P(S_t = j | x^T).

        Parameters
        ----------
        log_alphas
        log_betas

        Returns
        -------
        gamma : ndarray of shape (n_samples, n_states)
            Expectation of being in state j at time t given observations, P(S_t = j | x^T)
        """
        log_gamma = log_alphas + log_betas
        normalizer = logsumexp(log_gamma, axis=1, keepdims=True)
        log_gamma -= normalizer
        return log_gamma

    def compute_log_xi(self, log_alphas, log_betas):
        """Expected number of transitions from state i to j, (P(S_t-1 = j, S_t = i | x^T)"""
        # Initialize matrix of shape j X j
        # Number of expected transitions from state i to j
        log_xi = np.zeros(shape=(len(log_alphas)-1, self.n_states, self.n_states))

        with np.errstate(divide='ignore'):
            log_tpm = np.log(self.tpm)

        for i in range(self.n_states):
            for j in range(self.n_states):
                log_xi[:, i, j] = log_tpm[i, j] + log_alphas[:-1, i] + \
                                 log_betas[1:, j] + self.log_emission_probs_[1:, j]

        normalizer = logsumexp(log_xi, axis=(1,2))  # Take log sum over last two axes and keep first one.
        log_xi = log_xi - normalizer[:, np.newaxis, np.newaxis]
        return log_xi

    def _e_step(self, X: ndarray):
        '''
        Do a single e-step in Baum-Welch algorithm (Derives Xi and Gamma w.r.t. traditional HMM syntax)
        '''
        self.emission_probs(X)
        llk, log_alphas = self._log_forward_proba()
        log_betas = self._log_backward_proba()

        log_gamma = self.compute_log_posteriors(log_alphas, log_betas)  # Shape (n_samples, n_states)
        log_xi = self.compute_log_xi(log_alphas, log_betas)  # Shape (n_samples, n_states, n_states)

        """
        # Initialize matrix of shape j X j
        # Number of expected transitions from state i to j
        xi = np.zeros(shape=(self.n_states, self.n_states))
        for j in range(self.n_states):
            for k in range(self.n_states):
                xi[j, k] = self.tpm[j, k] * np.sum(
                    np.exp(log_alphas[:-1, j] + log_betas[1:, k] + self.log_emission_probs_[1:, k] - llk))
        """

        return log_gamma, log_xi, llk

    def _m_step(self, X: ndarray, log_gamma, log_xi):
        '''
        Given u and f do an m-step.
        Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        # Update transition matrix and initial probs
        #self.tpm = xi / np.sum(xi, axis=1).reshape((-1, 1))  # Check if this actually sums correct and to 1 on rows

        trans_probs = logsumexp(log_xi, axis=0) - logsumexp(log_gamma, axis=0).reshape(-1, 1)
        trans_probs = np.exp(trans_probs)
        self.tpm = trans_probs / np.sum(trans_probs, axis=1).reshape(-1,1)  # Make rows sum to 1

        gamma = np.exp(log_gamma)

        self.start_proba = gamma[0, :] / np.sum(gamma[0, :])

        # Update state-dependent distributions
        for j in range(self.n_states):
            self.mu[j] = np.sum(gamma[:, j] * X) / np.sum(gamma[:, j])
            self.std[j] = np.sqrt(np.sum(gamma[:, j] * np.square(X - self.mu[j])) / np.sum(gamma[:, j]))

    def _fit(self, X: ndarray, verbose=False):
        """ Container for looping part of fit-method"""
        self.old_llk = -np.inf  # Used to check model convergence
        self.best_epoch = -np.inf

        for epoch in range(self.epochs):
            # Do new init at each epoch
            self._init_params(X, output_hmm_params=True)

            for iter in range(self.max_iter):
                # Do e- and m-step
                log_gamma, log_xi, llk = self._e_step(X)
                self._m_step(X, log_gamma, log_xi)  # Updates mu, std and tpm

                # Check convergence criterion
                crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
                if crit < self.tol:
                    self.is_fitted = True
                    if llk > self.best_epoch:
                        self.best_epoch = llk
                        self.stationary_dist = self.get_stationary_dist(tpm=self.tpm)

                        self.best_tpm = self.tpm
                        self.best_start_proba = self.start_proba
                        self.best_mu = self.mu
                        self.best_std = self.std

                        if verbose == 2:
                            print(
                                f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - TPM {np.diag(self.tpm)} - Delta {self.start_proba}')
                    break
                else:
                    self.old_llk = llk

    def fit(self, X: ndarray, sort_state_seq=True, verbose=False):
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
        self._fit(X, verbose=verbose)

        if self.is_fitted is False:
            max_iter = self.max_iter
            self.max_iter = max_iter * 2  # Double amount of iterations
            self._fit(X)  # Try fitting again
            self.max_iter = max_iter  # Reset max_iter back to user-input
            if self.is_fitted == False and verbose == True:
                print(f'MLE NOT FITTED -- epochs {self.epochs} -- iters {self.max_iter*2} -- mu {self.mu} -- std {self.std} -- tpm {np.diag(self.tpm)}')

        if sort_state_seq is True:
            self._check_state_sort()  # Ensures low variance state is first

        self.tpm = self.best_tpm
        self.start_proba = self.best_start_proba
        self.mu = self.best_mu
        self.std = self.best_std

    def _check_state_sort(self):
        # Sort array ascending and check if order is changed
        # If the order is changed then states are reversed
        if np.sort(self.best_std)[0] != self.best_std[0]:
            # TODO only works for 2-states
            self.best_mu = self.best_mu[::-1]
            self.best_std = self.best_std[::-1]
            self.best_tpm = np.flip(self.best_tpm)
            self.best_start_proba = self.start_proba[::-1]


if __name__ == '__main__':
    sampler = SampleHMM(n_states=2, random_state=42)
    X, viterbi_states, true_states = sampler.sample_with_viterbi(1000, 1)
    model = GaussianHMM(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)

    model.fit(X, verbose=True)
    print(model.tpm.sum(axis=1))

    #pool = Pool()
    #mapfunc = partial(model.fit, verbose=False)
    #result = pool.map(mapfunc, [X]*20)