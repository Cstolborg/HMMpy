import numpy as np
from numpy import ndarray
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from utils.simulate_returns import simulate_2state_gaussian


class JumpHMM(BaseEstimator):

    def __init__(self, n_states: int = 2, max_iter: int = 100, tol: int = 1e-4,
                 epochs: int = 1, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.max_iter = max_iter  # Max iterations to fit model
        self.epochs = epochs  # Set number of random inits used in model fitting
        self.tol = tol

        # Init parameters initial distribution, transition matrix and state-dependent distributions from function
        np.random.seed(self.random_state)

    def construct_features(self, X: ndarray, window_len: int):
        N = len(X)
        Y = np.zeros(shape=(len(X), 9))
        df = pd.DataFrame(Y, columns=list('123456789'))
        Y[:, 0] = X  # Observations


        # Absolute changes
        Y[1:, 1] = np.abs(np.diff(X))
        Y[:-1, 2] = np.abs(np.diff(X))

        def right_mean(X, N):  # Consider using pandas rolling mean for this as welL (written)
            #R_m = np.cumsum(np.insert(X, 0, window_len)) # Cumulative sum for a given window
            #return (R_m[window_len:] - R_m[:-window_len]) / float(window_len)
            moving_mean = df.rolling(window_len).mean().shift(-2)
            return moving_mean

        x_1 = np.convolve(X, np.ones(N) / 3, mode='valid')

        def right_std(X, N):
            std_roll = df
            moving_std_right= std_roll.rolling(window_len).std(ddof=0).shift(-2) #note standard is 1 (sample)
            return moving_std_right

        def left_mean(X,N):
            #L_m = np.cumsum(np.insert(X, 0, window_len)) # Consider using pandas rolling mean for this as welL!
            #return (L_m[window_len:] - L_m[:-window_len]) / float(window_len)
            moving_mean_left = df.rolling(window_len).mean()
            return moving_mean_left

        def left_std(X,N):
            std_roll = df
            moving_std_left = std_roll.rolling(window_len).std(ddof=0)
            return moving_std_left



        # Centered moments
        #start_idx, stop_idx = int(window_len/2), -int(window_len/2)
        #Y[start_idx:stop_idx, 3]
        #vec = np.cumsum(np.insert(X, 0, 0))
        #print((vec[start_idx:] - vec[:-stop_idx]) / (window_len / 2 + 2) )
        #Y[(start_idx+1):(stop_idx-1), 3] = (vec[start_idx:] - vec[:-stop_idx]) / (window_len / 2 + 2)
        print(Y)
        print("Left absolute change = ",Y[1:, 1])
        print('Right absolute change =', Y[:-1, 2])
        print("Right mean = ", right_mean(X,N))
        print("Right std = ", right_std(X,N))
        print("Left mean = ", left_mean(X,N))
        print('Left std = ', left_std(X,N))



    def loss(self, X, theta):
        return np.sqrt(np.sum(np.square(X - theta)))

if __name__ == '__main__':
    model = JumpHMM(n_states=2)
    #returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions
    returns = np.arange(1,10,1)
    returns = returns ** 2
    model.construct_features(returns, window_len= 3)