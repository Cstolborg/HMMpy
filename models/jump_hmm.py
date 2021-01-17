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


        # Right local moments
        def right_mean(X, N):  # Consider using pandas rolling mean for this as welL (written)
            #R_m = np.cumsum(np.insert(X, 0, window_len)) # Cumulative sum for a given window
            #return (R_m[window_len:] - R_m[:-window_len]) / float(window_len)
            right_local_mean = df.rolling(window_len).mean().shift(-window_len+1)
            return right_local_mean

        def right_std(X, N):
            right_local_std = df.rolling(window_len).std(ddof=0).shift(-window_len+1)
            return right_local_std


        # Left local moments
        def left_mean(X,N):
            left_local_mean = df.rolling(window_len).mean()
            return left_local_mean

        def left_std(X,N):
            left_local_std = df.rolling(window_len).std(ddof=0)
            return left_local_std


        # Centered moments
        def local_mean (X,N):
            look_ahead = df.rolling(window_len).sum().shift(-window_len) # Looks forward with window_len
            look_back = df.rolling(window_len).sum() #Includes current position and looks window_len - 1 backward
            centered_local_mean = (look_ahead + look_back) / (2 * window_len)
            return centered_local_mean

        def local_std (X,N):
            centered_local_std = df.rolling(window_len*2).std(ddof = 0).shift(-window_len) #Rolls from 0 and 2x length iteratively, then shifts back 1x window length
            return centered_local_std

        # Prints used to validate calculations:

        #print(Y)
        #print("Left absolute change = ",Y[1:, 1])
        #print('Right absolute change =', Y[:-1, 2])
        #print("Right mean = ", right_mean(X,N))
        #print("Right std = ", right_std(X,N))
        #print("Left mean = ", left_mean(X,N))
        #print('Left std = ', left_std(X,N))
        #print("Local mean = ", local_mean(X,N))
        #print('Local std =', local_std(X,N))



    def loss(self, X, theta):
        return np.sqrt(np.sum(np.square(X - theta)))

if __name__ == '__main__':
    model = JumpHMM(n_states=2)
    #returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions
    returns = np.arange(1,10,1)
    returns = returns ** 2
    model.construct_features(returns, window_len= 3)