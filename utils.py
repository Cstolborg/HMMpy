import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#from base_hmm import BaseHiddenMarkov

""" TODO

We need to be able to sample from a HMM not just from two combined distributions.

Need a sampling function that is easily adapted to N states and any conditional dists.
  - Incl. t distributions.

"""

def simulate_2state_gaussian(N=200, means=None, std=None, plotting=False, random_state=42):
    '''Simulate data for bull and bear market using Normal distribution

    If means and std = None, the distributions follow those estimated by Hardy(2001)
    which are based on monthly returns from a stock index.
    '''

    if means == None:
        bull_mean = 0.1
        bear_mean = -0.05

    if std == None:
        bull_std = 0.1
        bear_std = 0.2

    np.random.seed(random_state)
    market_bull_1 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)
    market_bear_2 = stats.norm.rvs(loc=bear_mean, scale=bear_std, size=N)
    market_bull_3 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)
    market_bear_4 = stats.norm.rvs(loc=bear_mean, scale=bear_std, size=N)
    market_bull_5 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)

    # Create the list of true regime states and full returns list
    returns = np.array([market_bull_1] + [market_bear_2] + [market_bull_3] + [market_bear_4] + [market_bull_5]).flatten()
    true_regimes = np.array([np.zeros(N), np.ones(N), np.zeros(N), np.ones(N), np.zeros(N)]).flatten()

    #true_regimes = np.array([len(market_bull_1)] + [len(market_bear_2)] + [len(market_bull_3)] + [len(market_bear_4)] + [len(market_bull_5)]).flatten()

    if plotting:
        plt.plot(returns)
        plt.plot(true_regimes)
        plt.show()

    return returns, true_regimes




def simulate_bear_t(N=200, means=None, std=None, df=None, plotting=True, random_state=42):  ### Simulate 3 state.
        '''Simulate data for bull and bear market using Normal distribution

            If means and std = None, the distributions follow those estimated by Hardy(2001)
            which are based on monthly returns from a stock index.
        '''
        if means == None:
            bull_mean = 0.1
            bear_mean = -0.05

        if std == None:
            bull_std = 0.1
            bear_std = 0.2

        if df == None:
            bear_df = N - 1  # Bull state is only normal distribution hence it is not specified.

        np.random.seed(random_state)
        market_bull_1 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)
        market_bear_2 = stats.t.rvs(loc=bear_mean, scale=bear_std, size=N, df=bear_df)
        market_bull_3 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)
        market_bear_4 = stats.t.rvs(loc=bear_mean, scale=bear_std, size=N, df=bear_df)
        market_bull_5 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)

        returns = np.array([market_bull_1] + [market_bear_2] + [market_bull_3] + [market_bear_4] + [market_bull_5]).flatten()
        true_regimes = np.array([np.zeros(N), np.ones(N), np.zeros(N), np.ones(N), np.zeros(N)]).flatten()

        if plotting:
            plt.plot(returns)
            plt.plot(true_regimes)
            plt.show()

        return returns, true_regimes


def simulate_3state_gaussian_t(N=200, means=None, std=None, df=None, plotting=True, random_state=42):  # Simulate 3 state.
    if means == None:
        bull_mean = 0.1
        bear_mean = -0.05
        recession_mean = -0.10

    if std == None:
        bull_std = 0.1
        bear_std = 0.2
        recession_std = 0.3

    if df == None:
        recession_df = N - 1  # Bull and bear state is normal distribution hence it is not specified.



if __name__ == '__main__':
    returns, true_regimes = simulate_bear_t(plotting= True)
    print(true_regimes)

    model = BaseHiddenMarkov(2)
    model.fit(returns)
    model._viterbi(returns)




