import numpy as np
from numpy import ndarray
from scipy import stats
from scipy.special import logsumexp

from utils.simulate_returns import simulate_2state_gaussian
from utils.hmm_sampler import SampleHMM
from models.hidden_markov.hmm_base import BaseHiddenMarkov

from multiprocessing import Pool
from functools import partial


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
        gamma = log_alphas + log_betas
        normalizer = logsumexp(gamma, axis=1, keepdims=True)
        gamma -= normalizer
        return gamma

    def compute_log_xi(self, log_alphas, log_betas):
        """Expected number of transitions from state i to j, (P(S_t-1 = j, S_t = i | x^T)"""
        # Initialize matrix of shape j X j
        # Number of expected transitions from state i to j
        log_xi = np.zeros(shape=(len(log_alphas)-1, self.n_states, self.n_states))
        normalizer = np.zeros(shape=(self.n_states, self.n_states))
        log_tpm = np.log(self.tpm)
        for i in range(self.n_states):
            for j in range(self.n_states):
                log_xi[:, i, j] = log_tpm[i, j] + log_alphas[:-1, i] + \
                                 log_betas[1:, j] + self.log_emission_probs_[1:, j]

        normalizer = logsumexp(log_xi, axis=(1,2))  # Take log sum over last two axes and keep first one.
        xi = log_xi - normalizer[:, np.newaxis, np.newaxis]
        return xi

    def _e_step(self, X: ndarray):
        '''
        Do a single e-step in Baum-Welch algorithm (Derives Xi and Gamma w.r.t. traditional HMM syntax)
        '''
        self.emission_probs(X)
        llk, log_alphas = self._log_forward_proba()
        log_betas = self._log_backward_proba()

        gamma = self.compute_log_posteriors(log_alphas, log_betas)  # Shape (n_samples, n_states)
        xi = self.compute_log_xi(log_alphas, log_betas)  # Shape (n_samples, n_states, n_states)

        """
        # Initialize matrix of shape j X j
        # Number of expected transitions from state i to j
        xi = np.zeros(shape=(self.n_states, self.n_states))
        for j in range(self.n_states):
            for k in range(self.n_states):
                xi[j, k] = self.tpm[j, k] * np.sum(
                    np.exp(log_alphas[:-1, j] + log_betas[1:, k] + self.log_emission_probs_[1:, k] - llk))
        """

        return gamma, xi, llk

    def _m_step(self, X: ndarray, gamma, xi):
        '''
        Given u and f do an m-step.
        Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        # Update transition matrix and initial probs
        #self.tpm = xi / np.sum(xi, axis=1).reshape((-1, 1))  # Check if this actually sums correct and to 1 on rows

        self.tpm = np.exp(logsumexp(xi, axis=0) - logsumexp(gamma, axis=0).reshape(-1,1))

        gamma = np.exp(gamma)

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
                        self.best_epoch = llk
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
                                f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - TPM {np.diag(self.tpm)} - Delta {self.start_proba}')
                    break

                elif iter == self.max_iter - 1 and verbose == 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_llk = llk

        self.tpm = self.best_tpm
        self.start_proba = self.best_delta
        self.mu = self.best_mu
        self.std = self.best_std

class OnlineHMM(EMHiddenMarkov, BaseHiddenMarkov):

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: float = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

    def _init_posteriors(self, X, forget_fac):
        self.emission_probs(X)
        llk, log_alphas = self._log_forward_proba()
        log_betas = self._log_backward_proba()
        posteriors = self.compute_log_posteriors(log_alphas, log_betas)  # 2-D array (n_samples, n_states)

        multipliers = np.arange(1, len(X)+1, 1)[::-1]
        posteriors_exp = posteriors * multipliers[:, np.newaxis]

        return posteriors_exp

    def train(self, X, n_init_obs=250, forget_fac=0.9925):
        self._init_params()

        #posteriors_exp = self._init_posteriors(X[:250], forget_fac=forget_fac)
        #X = X[250:]

        self.log_forward_proba = np.zeros(shape=(len(X), self.n_states))
        self.posteriors = np.zeros(shape=(len(X), self.n_states))
        self.rec = np.zeros(shape=(len(X), self.n_states))

        self.log_forward_proba[0] = np.log(self.start_proba) + stats.norm.logpdf(X[0], loc=self.mu, scale=self.std)
        self.posteriors[0] = np.exp(self.log_forward_proba[0] - logsumexp(self.log_forward_proba[0]))
        #self.rec[0] = forget_fac * posteriors_exp[-1] + (1 - forget_fac) * self.posteriors[0]

        for t in range(1, len(X)):

            log_tpm = np.log(self.tpm)
            log_proba = stats.norm.logpdf(X[t], loc=self.mu, scale=self.std)
            self.log_forward_proba[t] = logsumexp((self.log_forward_proba[t - 1] + log_tpm.T).T, axis=0) + log_proba
            llk = logsumexp(self.log_forward_proba[t])

            self.posteriors[t] = np.exp(self.log_forward_proba[t] - llk)
            #self.rec[t] = forget_fac * self.rec[t - 1] + (1 - forget_fac) * self.posteriors[t]

            self.trans_proba = np.zeros((2,2))  # TODO move outside loop?
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.trans_proba[i,j] = self.log_forward_proba[t - 1, i] + log_tpm[i, j] \
                                            + log_proba[j] - llk

            self.trans_proba = np.exp(self.trans_proba)

            sum1 = np.sum(self.posteriors[1:t], axis=0)
            sum2 = np.sum(self.posteriors[1:t+1], axis=0)

            print(sum1)
            print(sum2)
            print(self.trans_proba)
            print('___')
            print(self.tpm)
            print(self.tpm.sum(axis=1))
            print('-'*40)

            if t > 2:
                self.tpm = (sum1 / sum2 * self.tpm.T).T \
                      (self.trans_proba.T / sum2).T
            else:
                self.tpm = (self.trans_proba.T / sum2).T

            #self.tpm = (self.rec[t-1] / self.rec[t] * self.tpm.T).T + (self.trans_proba.T / self.rec[t]).T
            #self.tpm = self.tpm / np.sum(self.tpm, axis=1).reshape((-1, 1))


if __name__ == '__main__':
    sampler = SampleHMM(n_states=2)
    X, viterbi_states, true_states = sampler.sample_with_viterbi(1000, 1)
    model = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)

    model.fit(X, verbose=True)
    print(model.tpm.sum(axis=1))

    #model = OnlineHMM(n_states=2, init='random', random_state=42)
    #model.train(X)

    #print(model.tpm)




    #pool = Pool()
    #mapfunc = partial(model.fit, verbose=False)
    #result = pool.map(mapfunc, [X]*20)