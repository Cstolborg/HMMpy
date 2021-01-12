import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from utils import simulate_2state_gaussian
from base_hmm import BaseHiddenMarkov

""" TODO

Move all state-dependent functions from the base class into this class

Enable modelling of conditional t distribution:
    - Requires the P(x), log_all_probs(X) to be updated
    - m-step must be updated

"""

class GaussianHMM(BaseHiddenMarkov):
    """
    Class for estimating gaussian HMMs.
    """

    def __init__(self, n_states: int = 2, random_state=42):
        super().__init__(n_states)
        np.random.seed(self.random_state)

    def P(self, x: int):
        """Function for computing diagonal prob matrix P(x).
         Change the function depending on the type of distribution you want to evaluate"""

        diag_probs = stats.norm.pdf(x, loc=self.mu, scale=self.std)  # Evalute x in every state
        diag_probs = np.diag(diag_probs)  # Transforms it into a diagonal matrix
        return diag_probs

    def emission_probs(self, X: list):
        """
        Compute all different probabilities p(x) given an observation sequence and n states

        Parameters
        ----------
        X: 1D-array
            Observation sequence

        Returns
        -------
        T X N matrix of emission probabilities
        """
        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init T X N matrix
        probs = np.zeros((T, self.n_states))

        # For all states evaluate the density function
        for j in range(self.n_states):
            log_probs[:, j] = stats.norm.logpdf(X, loc=self.mu[j], scale=self.std[j])

        probs = np.exp(log_probs)

        return probs, log_probs

    def _m_step(self, X, u, f):
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

    def fit(self, X, verbose=0):
        """Iterates through the e-step and the m-step"""

        for iter in range(self.max_iter):
            u, f, llk = self._e_step(X)
            self._m_step(X, u, f)

            if verbose == 1:
                print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {self.T.flatten()} - Delta {self.delta}')

            # Check convergence
            crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
            if crit < self.tol:
                print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {self.T.flatten()} - Delta {self.delta}')
                break

            elif iter == self.max_iter-1:
                print(f'No convergence after {iter} iterations')

            else:
                self.old_llk = llk

    def predict(self, X):
        state_preds, posteriors = self._viterbi(X)
        return state_preds, posteriors


if __name__ == '__main__':

    model = GaussianHMM(n_states=2)
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
