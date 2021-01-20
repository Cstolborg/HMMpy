import numpy as np
from numpy import ndarray
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.cluster._kmeans import kmeans_plusplus

import matplotlib.pyplot as plt

from utils.simulate_returns import simulate_2state_gaussian


class JumpHMM(BaseEstimator):

    def __init__(self, n_states: int = 2, jump_penalty: float = .2, init: str = 'kmeans++',
                 max_iter: int = 30, tol: int = 1e-4,
                 epochs: int = 10, random_state: int = 42):
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.random_state = random_state
        self.max_iter = max_iter  # Max iterations to fit model
        self.epochs = epochs  # Set number of random inits used in model fitting
        self.tol = tol
        self.init = init
        self.best_objective_likelihood = np.inf

        # Init parameters initial distribution, transition matrix and state-dependent distributions from function
        np.random.seed(self.random_state)

    def construct_features(self, X: ndarray, window_len: int):
        N = len(X)
        df = pd.DataFrame(X)

        df['Left local mean'] = df.rolling(window_len).mean()
        df['Left local std'] = df[0].rolling(window_len).std(ddof=0)

        df['Right local mean'] = df[0].rolling(window_len).mean().shift(-window_len+1)
        df['Right local std'] = df[0].rolling(window_len).std(ddof=0).shift(-window_len + 1)

        look_ahead = df[0].rolling(window_len).sum().shift(-window_len)  # Looks forward with window_len (Helper 1)
        look_back = df[0].rolling(window_len).sum()  # Includes current position and looks window_len - 1 backward (Helper 2)
        df['Central local mean'] = (look_ahead + look_back) / (2 * window_len)
        df['Centered local std'] = df[0].rolling(window_len * 2).std(ddof=0).shift(-window_len)  # Rolls from 0 and 2x length iteratively, then shifts back 1x window length

        # Absolute changes
        #Y[1:, 1] = np.abs(np.diff(X))
        #Y[:-1, 2] = np.abs(np.diff(X))
        Z = df.dropna().to_numpy()
        self.n_features = Z.shape[1]

        return Z

    def _init_params(self, Z: ndarray):


        if self.init == 'kmeans++':
            # Theta
            centroids, _ = kmeans_plusplus(Z, self.n_states)  # Use sklearns kmeans++ algorithm
            self.theta = centroids.T  # Transpose to get shape: N_features X N_states

            # State sequence
            self._fit_state_seq(Z)  # this function implicitly updates self.state_seq

        else:
            # Init theta as zeros and sample state seq from uniform dist
            self.theta = np.zeros(shape=(self.n_features, self.n_states))  # Init as empty matrix
            state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=int)  # Array of possible states
            state_seq = np.random.choice(a=state_index, size=len(Z))  # Sample sequence uniformly
            self.state_seq = state_seq

        self.old_state_seq = self.state_seq  # Required in fit() method

    def _fit_theta(self, Z):
        for j in range(self.n_states):
            state_slicer = self.state_seq == j  # Boolean array of True/False

            N_j = sum(state_slicer)  # Number of terms in state j
            z = Z[state_slicer].sum(axis=0)  # Sum each feature across time
            self.theta[:, j] = 1 / N_j * z

    def _fit_state_seq(self, Z: ndarray):
        T = len(Z)

        losses = np.zeros(shape=(T, self.n_states))  # Init T X N matrix
        losses[-1, :] = self.loss(Z[-1], self.theta)  # loss corresponding to last state

        # Do a backward recursion to get losses
        for t in range(T - 2, -1, -1):  # Count backwards
            current_loss = self.loss(Z[t], self.theta)  # n-state vector of current losses
            last_loss = self.loss(Z[t + 1], self.theta)  # n-state vector of last losses

            for j in range(self.n_states):  # TODO redefine this as a matrix problem and get rid of loop
                state_change_penalty = np.ones(self.n_states) * self.jump_penalty  # Init jump penalty
                state_change_penalty[j] = 0  # And remove it for all but current state

                losses[t, j] = current_loss[j] + np.min(last_loss + state_change_penalty)

        # From losses get the most likely sequence of states i
        state_preds = np.zeros(T).astype(int)  # Vector of length N
        state_preds[0] = np.argmin(losses[0])  # First most likely state is the index position

        # Do a forward recursion to calculate most likely state sequence
        for t in range(1, T):  # Count backwards
            last_state = state_preds[t-1]

            state_change_penalty = np.ones(self.n_states) * self.jump_penalty  # Init jump penalty
            state_change_penalty[last_state] = 0  # And remove it for all but current state

            state_preds[t] = np.argmin(losses[t] + state_change_penalty)

        # Finally compute score of objective function
        all_likelihoods = losses[np.arange(len(losses)), state_preds].sum()
        jump_penalty = np.diff(state_preds) != 0   # True/False array showing state changes
        jump_penalty = (jump_penalty * self.jump_penalty).sum()  # Multiply all True values with penalty


        self.objective_likelihood = all_likelihoods + jump_penalty
        self.state_seq = state_preds


    def fit(self, Z: ndarray, verbose=0):
        # Init params
        self._init_params(Z)

        for epoch in range(self.epochs):
            # Do new init at each epoch
            if epoch > 1: self._init_params(Z)

            for iter in range(self.max_iter):
                self._fit_theta(Z)
                self._fit_state_seq(Z)

                # Check convergence criterion
                crit = np.array_equal(self.state_seq, self.old_state_seq)  # Improvement in log likelihood
                if crit == True:
                    if self.objective_likelihood < self.best_objective_likelihood:
                        self.best_objective_likelihood = self.objective_likelihood
                        self.best_sequence = self.state_seq
                        self.best_theta = self.theta

                    print(f'Epoch {epoch} -- Iter {iter} -- likelihood {self.objective_likelihood} -- Theta {self.theta[0]} ')#Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')
                    break

                elif iter == self.max_iter - 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_state_seq = self.state_seq

    def loss(self, z, theta): # z must always be a vector but theta can be either a vector or a matrix
        # Subtract z from theta row-wise. Requires the transpose of the column matrix theta
        diff = (theta.T - z).T
        return np.linalg.norm(diff, axis=0) ** 2  # squared l2 norm.

    def state_change_indicator(self):  # TODO deprecate this
        # Take the diff over the array. If it is different than zero a state change has occurred
        indicator = np.diff(self.state_seq) != 0
        indicator = np.append([False] , indicator)  # Add False as the first term since np.diff removes the first obs.
        return indicator


if __name__ == '__main__':
    model = JumpHMM(n_states=2, random_state=42)
    returns, true_regimes = simulate_2state_gaussian(
        plotting=False)  # Simulate some data from two normal distributions
    #returns = np.arange(1,20,1)
    #returns = returns ** 2

    Z = model.construct_features(returns, window_len=6)

    #model._init_params(Z)
    #model.state_seq = true_regimes[2:-3].astype(int)  # Arbitrarily give it the answer alrady


    model.fit(Z)


    plotting = False
    if plotting == True:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        #ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
        #ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
        #ax[1].plot(first_seq, label='First Predicted states', ls='dashdot')
        ax[1].plot(model.state_seq, label='Predicted states', ls='dotted')
        ax[1].plot(true_regimes, label='True states', ls='dashed')

        plt.legend()
        plt.show()


'''
    print(Z[0], Z[0].shape)
    print(Z[0, None], Z[0, None].shape)
    print(model.theta, model.theta.shape)
    print('.'*40)
    print(model.loss(Z[0], model.theta ))
'''




