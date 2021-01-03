import numpy as np
from scipy import stats

def simulate_obs(N=150, means=None, std=None, plotting=False, random_state=42):
    '''Simulate data for bull and bear market using Normal distribution '''
    import matplotlib.pyplot as plt
    if means == None:
        bull_mean = 0.1
        bear_mean = -0.05

    if std == None:
        bull_std = np.sqrt(0.1)
        bear_std = np.sqrt(0.2)

    np.random.seed(random_state)
    market_bull_1 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)
    market_bear_2 = stats.norm.rvs(loc=bear_mean, scale=bear_std, size=N)
    market_bull_3 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)
    market_bear_4 = stats.norm.rvs(loc=bear_mean, scale=bear_std, size=N)
    market_bull_5 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)

    obs = np.array([market_bull_1]+ [market_bear_2]+ [market_bull_3]+ [market_bear_4]+ [market_bull_5]).flatten()

    if plotting:
        plt.plot(obs)
        plt.show()

    return obs