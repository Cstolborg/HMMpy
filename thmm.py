import numpy as np
from scipy import stats
from scipy.special import digamma
import matplotlib.pyplot as plt

from utils import simulate_2state_gaussian
from base_hmm import BaseHiddenMarkov

import scipy.optimize as opt

""" TODO

Move all state-dependent functions from the base class into this class

Enable modelling of conditional t distribution:
    - Requires the P(x), log_all_probs(X) to be updated
    - m-step must be updated

"""

import scipy.optimize as opt


class GaussianHMM(BaseHiddenMarkov):
    """
    Class for estimating gaussian HMMs.
    """

    def __init__(self, n_states, random_state=42):
        super().__init__(n_states)
        self.dof = 5

        # Random init of state distributions
        self.random_state = random_state
        np.random.seed(self.random_state)

    def P(self, x: int):
        """Function for computing diagonal prob matrix P(x).
         Change the function depending on the type of distribution you want to evaluate"""

        diag_probs = stats.t.pdf(x, loc=self.mu, scale=self.std, df=self.dof)  # Evalute x in every state
        diag_probs = np.diag(diag_probs)  # Transforms it into a diagonal matrix
        return diag_probs

    def emission_probs(self, X: list):
        """ Compute all different log probabilities log(p(x)) given an observation sequence and n states

        Returns: T X N matrix
        """
        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init N X M matrix
        probs = np.zeros((T, self.n_states))

        # For all states evaluate the density function
        for j in range(self.n_states):
            log_probs[:, j] = stats.t.logpdf(X, loc=self.mu[j], scale=self.std[j], df=self.dof)

        probs = np.exp(log_probs)
        return probs, log_probs

    def _m_step(self, X, u, f, iterations):
        ''' Given u and f do an m-step.
          Update degrees of freedom iteratively.
         Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        X = np.array(X)
        T = len(X)

        # Empty list to store u_it values
        self.u_it = np.zeros((T, self.n_states))  # Init N X M matrix

        # Instantiating the u_it
        self.u_it[0, :] = (np.square(self.std) * (self.dof + 1) / (np.square(self.std) * self.dof + np.square((X[0] - self.mu))))
        # df to be updated to formula on page 34 in Nystrup Master Thesis

        # Update transition matrix and initial probs
        self.T = f / np.sum(f, axis=1).reshape((2, 1))  # Check if this actually sums correct and to 1 on rows
        self.delta = u[0, :] / np.sum(u[0, :])


        # Update state-dependent distributions
        for iteration in range(iterations):
            for j in range(self.n_states):
                if j == 0:
                    self.mu[j] = np.sum(u[:, j] * X * self.u_it[:, j]) / (np.sum(u[:, j]*self.u_it[:, j]))
                    self.std[j] = np.sqrt(np.sum(u[:, j] * np.square(X - self.mu[j])) / np.sum(u[:, j]))
                    self.u_it[:, j] = np.square(self.std[j]) * (self.dof + 1) / ((np.square(self.std[j]) * self.dof) + np.square(X - self.mu[j]))

                    # Find root of some estimator function based on digammas and dof
                    def dof_estimator(dof):  # TODO research why Nystrup master p. 94 use log(dof) rather than dof
                        term1 = 1-digamma(0.5*dof)
                        term2 = np.log(0.5*dof)
                        term3 = digamma((dof+1)/2)
                        term4 = -np.log((dof+1)/2)
                        term5 = 1 / np.sum(u[:, j]) * np.sum(u[:, j] * (np.log(self.u_it[:, j])-self.u_it[:, j]))
                        return term1 + term2 + term3 + term4 + term5

                    self.dof = opt.root(fun=dof_estimator, x0=self.dof).x


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

    model = GaussianHMM(n_states=2)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some X in two states from normal distributions

    #model.fit(returns, verbose=0)
    u, f, llk = model._e_step(returns)
    model._m_step(returns, u, f, iterations=20)

    #print("u_it = ", model.u_it)
    #print("mu =", model.mu)
    #print("std = ",model.std)


    #states, posteriors = model.predict(returns)

