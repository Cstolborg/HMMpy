import numpy as np
from scipy import stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from hmmpy.sampler import SampleHMM
from hmmpy.base import BaseHiddenMarkov
from hmmpy.mle import MLEHMM


class OnlineHMM(MLEHMM, BaseHiddenMarkov):

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: float = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

    def _init_posteriors(self, X, forget_fac):
        self.emission_probs(X)
        llk, log_alphas = self._log_forward_proba()
        log_betas = self._log_backward_proba()
        log_posteriors = self.compute_log_posteriors(log_alphas, log_betas)  # 2-D array (n_samples, n_states)

        forget_factors = forget_fac**(np.arange(1, len(X)+1, 1)[::-1])
        posteriors = np.exp(log_posteriors) * forget_factors[:, np.newaxis]

        return posteriors

    def train_no_expsmoothing(self, X, n_init_obs=250, forget_fac=0.9925):
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
            self.log_forward_proba[t] = logsumexp(log_tpm + self.log_forward_proba[t - 1], axis=0) + log_proba
            llk = logsumexp(self.log_forward_proba[t])

            self.posteriors[t] = np.exp(self.log_forward_proba[t] - llk)
            #self.rec[t] = forget_fac * self.rec[t - 1] + (1 - forget_fac) * self.posteriors[t]

            log_xi = np.zeros((2, 2))  # TODO move outside loop?
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[i, j] = self.log_forward_proba[t - 1, i] + log_tpm[i, j] \
                                        + log_proba[j]

            normalizer = logsumexp(log_xi)
            log_xi = log_xi - normalizer
            xi = np.exp(log_xi)

            # TODO inefficient to sum everything in each loop
            sum1 = np.sum(self.posteriors[1:t], axis=0)
            sum2 = np.sum(self.posteriors[1:t+1], axis=0)

            if t > 2:
                self.tpm = (sum1 / sum2).reshape(-1, 1) * self.tpm \
                      + xi / sum2.reshape(-1, 1)

            self.tpm = self.tpm / self.tpm.sum(axis=1).reshape(-1, 1)

            print(self.tpm)
            print('-'*40)

            #self.tpm = (self.rec[t-1] / self.rec[t] * self.tpm.T).T + (self.trans_proba.T / self.rec[t]).T
            #self.tpm = self.tpm / np.sum(self.tpm, axis=1).reshape((-1, 1))

    def train(self, X, n_init_obs=500, forget_fac=0.9925):
        # Initialize mu, std and tpm
        self._init_params()

        # Set mu and std to their real values to test functionality
        self.mu = np.array([100, 200])
        self.std = np.array([10, 10])

        #prev_posteriors = self._init_posteriors(X[:n_init_obs], forget_fac=forget_fac)
        #X = X[n_init_obs:]

        # Init empty arrays for storing values
        self.log_forward_proba = np.zeros(shape=(len(X), self.n_states))
        self.posteriors = np.zeros(shape=(len(X), self.n_states))
        self.rec = np.zeros(shape=(len(X), self.n_states))  # Exponential forgetting factor

        # Compute initial values before entering loop
        self.log_forward_proba[0] = np.log(self.start_proba) + stats.norm.logpdf(X[0], loc=self.mu, scale=self.std)
        self.posteriors[0] = np.exp(self.log_forward_proba[0] - logsumexp(self.log_forward_proba[0]))
        self.rec[0] =  (1 - forget_fac) * self.posteriors[0]

        # Arrays for storing values used for plotting later
        tpm = np.zeros(shape=(len(X), self.n_states))
        mu = np.zeros(shape=(len(X), self.n_states))
        std = np.zeros(shape=(len(X), self.n_states))
        tpm[0] = np.diag(self.tpm)
        mu[0] = self.mu
        std[0] = self.std

        for t in range(1, len(X)):
            # Compute log forward probabilities and log-likelihood
            log_tpm = np.log(self.tpm)
            log_proba = stats.norm.logpdf(X[t], loc=self.mu, scale=self.std)
            self.log_forward_proba[t] = logsumexp(log_tpm + self.log_forward_proba[t - 1], axis=0) + log_proba
            llk = logsumexp(self.log_forward_proba[t])

            # Exponential transformation to get posterior and its current sum
            self.posteriors[t] = np.exp(self.log_forward_proba[t] - llk)
            self.rec[t] = forget_fac * self.rec[t - 1] + (1 - forget_fac) * self.posteriors[t]

            # Compute log probability of state shift at time t
            log_xi = np.zeros((2, 2))  # TODO move outside loop?
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[i, j] = self.log_forward_proba[t - 1, i] + log_tpm[i, j] \
                                   + log_proba[j] - llk
            xi = np.exp(log_xi)

            # Compute tpm, mu and std based on Stenger (2001) formulas
            forget_frac = self.rec[t-1] / self.rec[t]
            self.tpm = forget_frac.reshape(-1, 1) * self.tpm \
                       + xi / self.rec[t].reshape(-1, 1)
            self.tpm = self.tpm / self.tpm.sum(axis=1).reshape(-1, 1)  # Make rows sum to 1
            self.mu = forget_frac * self.mu + self.posteriors[t] * X[t] / self.rec[t]
            var = forget_frac * np.square(self.std) + self.posteriors[t] * (X[t]-self.mu)**2 / self.rec[t]
            self.std = np.sqrt(var)

            tpm[t] = np.diag(self.tpm)
            mu[t] = self.mu
            std[t] = self.std
        return tpm, mu, std

if __name__ == '__main__':
    # Set hmm params equal to those of Stenger (2001)
    hmm_params = {'mu': np.array([100, 200]) ,
                  'std': np.sqrt(np.array([100, 100])),
                  'tpm': np.array([[0.9, 0.1],  # TODO figure out powers of vectors in python
                                   [0.2, 0.8]])
                  }
    # Sample observations from given hmm
    sampler = SampleHMM(n_states=2, random_state=42, hmm_params=hmm_params)
    X, viterbi_states, true_states = sampler.sample_with_viterbi(1000, 1)

    # Train online model
    model = OnlineHMM(n_states=2, init='random', random_state=42)
    tpm, mu, std = model.train(X, forget_fac=0.99)

    print(
        f'Means: {model.mu} - STD {model.std} - TPM {np.diag(model.tpm)} ')

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12,7), sharex=True)
    ax[0, 0].plot(tpm[:, 0], label='P(1)', )
    ax[0, 0].plot([sampler.tpm[0,0]]*len(tpm), label='True P1', ls='dashed')
    ax[0, 1].plot(tpm[:, 1], label='P(2)', )
    ax[0, 1].plot([sampler.tpm[1,1]]*len(tpm), label='True P2', ls='dashed')
    #ax[0].legend()

    ax[1, 0].plot(mu[:, 0])
    ax[1, 0].plot([sampler.mu[0]]*len(mu), ls='dashed')
    ax[1, 1].plot(mu[:, 1])
    ax[1, 1].plot([sampler.mu[1]] * len(mu), ls='dashed')

    ax[2, 0].plot(std[:, 0])  # label='True states', ls='dashed')
    ax[2, 0].plot([sampler.std[0]] * len(std), ls='dashed')
    ax[2, 1].plot(std[:, 1])  # label='True states', ls='dashed')
    ax[2, 1].plot([sampler.std[1]] * len(std), ls='dashed')

    ax[3, 0].plot(model.rec[:, 0])
    ax[3, 1].plot(model.rec[:, 1])

    plt.show()

    from hmmpy.utils.plotting.plot_hmm import plot_samples_states
    plot_samples_states(X, true_states)