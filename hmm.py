import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from utils import simulate_2state_gaussian
from Main import HiddenMarkovModel

""" TODO

Enable modelling of conditional t distribution:
    - Requires the P(x), log_all_probs(X) to be updated
    - m-step must be updated

"""


class HMM(HiddenMarkovModel):

    def __init__(self, n_states, random_state=42):
        super().__init__(n_states)

        self.delta = np.array([0.2, 0.8])  # 1 X N vector
        self.T = self._init_params()  # N X N transmission matrix

        # Random init of state distributions
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.mu = np.random.rand(n_states)
        self.std = np.random.rand(n_states)

    def _init_params(self):
        T = np.zeros((2, 2))
        T[0, 0] = 0.7
        T[0, 1] = 0.3
        T[1, 0] = 0.2
        T[1, 1] = 0.8
        return T

    def P(self, x: int):
        """Function for computing diagonal prob matrix P(x).
         Change the function depending on the type of distribution you want to evaluate"""

        diag_probs = stats.norm.pdf(x, loc=self.mu, scale=self.std)  # Evalute x in every state
        diag_probs = np.diag(diag_probs)  # Transforms it into a diagonal matrix
        return diag_probs

    def log_all_probs(self, X: list):
        """ Compute all different log probabilities log(p(x)) given an observation sequence and n states

        Returns: T X N matrix
        """
        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init N X M matrix

        # For all states evaluate the density function
        for j in range(self.n_states):
            log_probs[:, j] = stats.norm.logpdf(X, loc=self.mu[j], scale=self.std[j])

        return log_probs

    def fit(self, X, verbose=0):
        """Iterates through the e-step and the m-step"""

        for iter in range(self.epochs):
            u, f, llk = self._e_step(X)
            self._m_step(X, u, f)

            if verbose == 2:
                print(iter)
                print('MEAN: ', self.mu)
                print('STD: ', self.std)
                print('Gamma: ', self.T)
                print('DELTA', self.delta)
                print('loglikelihood', llk)

                print('.' * 40)

            if verbose == 1:
                print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {self.T.flatten()} - Delta {self.delta}')

            # Check convergence
            crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
            if crit < self.tol:
                print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {self.T.flatten()} - Delta {self.delta}')
                break

            elif iter == self.epochs-1:
                print(f'No convergence after {iter} iterations')

            else:
                self.old_llk = llk

    def predict(self, X):
        state_preds, posteriors = self._viterbi(X)
        return state_preds, posteriors


if __name__ == '__main__':

    model = HMM(n_states=2)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some X in two states from normal distributions

    model.fit(returns, verbose=0)

    states, posteriors = model.predict(returns)

    plotting = True
    if plotting == True:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
        ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
        ax[1].plot(states, label='Predicted states', ls='dotted')
        ax[1].plot(true_regimes, label='True states', ls='dashed')

        plt.legend()
        plt.show()
