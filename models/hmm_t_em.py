import numpy as np
from scipy import stats
from scipy.special import digamma
import scipy.optimize as opt
import matplotlib.pyplot as plt

from typing import List

from utils.simulate_returns import simulate_2state_gaussian
from hmm_gaussian_em import EMHiddenMarkov



class EMTHiddenMarkov(EMHiddenMarkov):
    """
    Class for estimating HMMs with a mixture of gaussian and students t distributions.
    """

    def __init__(self, n_states: int = 2, init: str = 'random', max_iter: int = 100, tol: int = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)
        self.dof = 2  # TODO how do we init this???

    def emission_probs(self, X: list):
        """ Compute all different log probabilities log(p(x)) given an observation sequence and n states

        Returns: T X N matrix
        """
        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init N X M matrix
        probs = np.zeros((T, self.n_states))

        # For all states evaluate the density function
        # In the last state, i.e. the high variance state use a t-dist
        for j in range(self.n_states):
            if j < (self.n_states-1):
                log_probs[:, j] = stats.norm.logpdf(X, loc=self.mu[j], scale=self.std[j])
            elif j == (self.n_states-1):
                log_probs[:, j] = stats.t.logpdf(X, loc=self.mu[j], scale=self.std[j], df=self.dof)

        probs = np.exp(log_probs)

        return probs, log_probs

    def _m_step(self, X , u, f, iterations: int = 2):
        ''' Given u and f do an m-step.
          Update degrees of freedom iteratively.
         Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        X = np.array(X)
        T = len(X)

        # init u_it
        u_it = np.square(self.std[-1]) * (self.dof + 1) / ((np.square(self.std[-1]) * self.dof) + np.square(X - self.mu[-1]))

        # Update transition matrix and initial probs
        self.T = f / np.sum(f, axis=1).reshape((-1, 1))  # Check if this actually sums correct and to 1 on rows
        self.delta = u[0, :] / np.sum(u[0, :])

        # TODO Remove iterations from findinf DOF as it is not necessary.
        # Update state-dependent distributions
        for iteration in range(iterations): # Iterate over this procedure until dof has converged
            for j in range(self.n_states):
                if j < (self.n_states-1):
                    self.mu[j] = np.sum(u[:, j] * X) / np.sum(u[:, j])
                    self.std[j] = np.sqrt(np.sum(u[:, j] * np.square(X - self.mu[j])) / np.sum(u[:, j]))

                if j == (self.n_states-1):
                    self.mu[j] = np.sum(u[:, j] * X * u_it) / (np.sum(u[:, j]*u_it))
                    self.std[j] = np.sqrt(np.sum(u[:, j] * np.square(X - self.mu[j])) / np.sum(u[:, j]))
                    self.u_it = np.square(self.std[j]) * (self.dof + 1) / ((np.square(self.std[j]) * self.dof) + np.square(X - self.mu[j]))

                    # Find root of some estimator function based on digammas and dof
                    def dof_estimator(dof):  # TODO research why Nystrup master p. 94 use exp(dof) rather than dof
                        term1 = 1 - digamma(0.5 * dof)
                        term2 = np.log(0.5 * dof)
                        term3 = digamma((dof + 1) / 2)
                        term4 = -np.log((dof + 1) / 2)
                        term5 = 1 / np.sum(u[:, j]) * np.sum(u[:, j] * (np.log(u_it) - u_it))
                        return term1 + term2 + term3 + term4 + term5

                    self.dof = opt.root(fun=dof_estimator, x0=self.dof).x




if __name__ == '__main__':
    model = EMTHiddenMarkov(n_states=2, epochs=5)
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
