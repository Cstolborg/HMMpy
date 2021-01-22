import numpy as np
from numpy import ndarray
from scipy import stats
import scipy.optimize as opt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster._kmeans import kmeans_plusplus

import matplotlib.pyplot as plt

from utils.simulate_returns import simulate_2state_gaussian


''' TODO:

Add predict func -- USE Zuchinni
add stationary distribution
 
implement simulation and BAC to choose jump penalty.

Z-score standardisation

'''

class JumpHMM(BaseEstimator):

    def __init__(self, n_states: int = 2, jump_penalty: float = .2, init: str = 'kmeans++',
                 max_iter: int = 30, tol: int = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.random_state = random_state
        self.max_iter = max_iter  # Max iterations to fit model
        self.epochs = epochs  # Set number of random inits used in model fitting
        self.tol = tol
        self.init = init
        self.best_objective_likelihood = np.inf
        self.old_objective_likelihood = np.inf

        # Init parameters initial distribution, transition matrix and state-dependent distributions from function
        np.random.seed(self.random_state)

    def construct_features(self, X: ndarray, window_len: int):  # TODO remove forward-looking params and slice X accordingly
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
        self.window_len = window_len

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
        state_changes = np.diff(state_preds) != 0   # True/False array showing state changes
        jump_penalty = (state_changes * self.jump_penalty).sum()  # Multiply all True values with penalty

        self.objective_likelihood = all_likelihoods + jump_penalty  # Float
        self.state_seq = state_preds

    def get_hmm_params(self, X: ndarray):  # TODO remove forward-looking params and slice X accordingly
        # Slice data
        if X.ndim == 1:  # Makes function compatible on Z
            X = X[(self.window_len-1) : -self.window_len]
        elif X.ndim > 1:
            X = X[:, 0]

        # group by states
        diff = np.diff(self.best_state_seq)
        df_states = pd.DataFrame({'state_seq': self.best_state_seq,
                                  'X': X,
                                  'state_sojourns': np.append([False], diff == 0),
                                  'state_changes': np.append([False], diff != 0)})

        state_groupby = df_states.groupby('state_seq')

        # Transition probabilities
        # TODO only works for a 2-state HMM
        self.T = np.diag(state_groupby['state_sojourns'].sum())
        state_changes = state_groupby['state_changes'].sum()
        self.T[0, 1] = state_changes[0]
        self.T[1, 0] = state_changes[1]
        self.T = self.T / self.T.sum(axis=1).reshape(-1, 1)  # make rows sum to 1

        # Conditional distributions
        self.mu = state_groupby['X'].mean().values.T  # transform mean back into 1darray
        self.std = state_groupby['X'].std().values.T

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
                crit1 = np.array_equal(self.state_seq, self.old_state_seq)  # No change in state sequence
                crit2 = np.abs(self.old_objective_likelihood - self.objective_likelihood)
                if crit1 == True or crit2 < self.tol:
                    if self.objective_likelihood < self.best_objective_likelihood:
                        self.best_objective_likelihood = self.objective_likelihood
                        self.best_state_seq = self.state_seq
                        self.best_theta = self.theta
                        self.get_hmm_params(Z)

                    print(f'Epoch {epoch} -- Iter {iter} -- likelihood {self.objective_likelihood} -- Theta {self.theta[0]} ')#Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')
                    break

                elif iter == self.max_iter - 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_state_seq = self.state_seq
                    self.old_objective_likelihood = self.objective_likelihood

    def loss(self, z, theta): # z must always be a vector but theta can be either a vector or a matrix
        # Subtract z from theta row-wise. Requires the transpose of the column matrix theta
        diff = (theta.T - z).T
        return np.linalg.norm(diff, axis=0) ** 2  # squared l2 norm.

    def sample(self, n_samples: int):
        '''
        Sample from a fitted hmm.

        Parameters
        ----------
        n_samples: int
                Amount of samples to generate

        Returns
        -------
        Sample of same size n_samples
        '''
        state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=int)  # Array of possible states
        sample_states = np.zeros(n_samples).astype(int)  # Init sample vector
        sample_states[0] = np.random.choice(a=state_index, size=1) # TODO add stationary distribution First state is determined by initial dist

        for t in range(1, n_samples):
            # Each new state is chosen using the transition probs corresponding to the previous state sojourn.
            sample_states[t] = np.random.choice(a=state_index, size=1, p=self.T[sample_states[t-1], :])

        samples = stats.norm.rvs(loc=self.mu[sample_states], scale = self.std[sample_states], size=n_samples)

        return samples, sample_states

    def get_stationary_dist(self):
        ones = np.ones(shape=(self.n_states,self.n_states))
        identity = np.diag(ones)
        init_guess = np.ones(self.n_states)/self.n_states

        def solve_stationary(stationary_dist):
            return (stationary_dist @ (identity - self.T + ones)) - np.ones(self.n_states)

        stationary_dist = opt.root(solve_stationary, x0=init_guess)
        print(stationary_dist)





if __name__ == '__main__':
    model = JumpHMM(n_states=2, random_state=42)
    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions

    Z = model.construct_features(returns, window_len=6)

    model.fit(Z)
    print(model.T)
    model.get_stationary_dist()

    plotting = False
    if plotting == True:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        #ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
        #ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
        #ax[1].plot(first_seq, label='First Predicted states', ls='dashdot')
        ax[1].plot(model.best_state_seq, label='Predicted states', ls='dotted')
        ax[1].plot(true_regimes, label='True states', ls='dashed')

        plt.legend()
        plt.show()


