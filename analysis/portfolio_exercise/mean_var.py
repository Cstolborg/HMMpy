import warnings

import numpy as np
import pandas as pd
import scipy.optimize as opt

from hmmpy.finance.backtest import Backtester
from hmmpy.finance.backtest import FinanceHMM
from hmmpy.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from hmmpy.hidden_markov.hmm_jump import JumpHMM
from hmmpy.utils.data_prep import DataPrep

np.seterr(divide='ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)


def mean_var(data):
    '''
    Outputs the following variables:
    weight_state, mu, std, mu_tan ,std_tan, mu_mv, std_mv, mu_C, std_C
    '''
    logrets = data.logrets.iloc[1000:, :-1]
    N_assets = len(logrets.columns)

    log_mu, log_cov = logrets.mean(), logrets.cov()
    mu, cov = FinanceHMM.logcov_to_cov(log_mu, log_cov)

    mu = (1 + logrets.mean()) ** 252 - 1  # Convert from daily to annualized returns
    Sigma = cov * 252

    ### Compute Tangency portfolio #############

    #### NUMERICAL SOLUTION TO TANGENCY
    def target_fun(w):
        return -(w.T.dot(mu) / np.sqrt(w.T.dot(Sigma).dot(w)))  # Maximize this

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},  # weights, sum to 1
            {"type": "ineq", "fun": lambda w: w},  # weights larger than 0
            {"type": "ineq", "fun": lambda w: -(np.abs(w)-0.4)})

    start_val = mu / sum(mu)

    print(opt.minimize(target_fun, x0=start_val, constraints=cons
                         ))
    w_tan = opt.minimize(target_fun, x0=start_val, constraints=cons
                         ).x

    w_tan = np.append(w_tan, 0)
    w_tan = pd.Series(w_tan, index=data.logrets.columns)

    return w_tan


if __name__ == '__main__':
    data = DataPrep(out_of_sample=True)

    w_tan = mean_var(data)
    print(w_tan.round(3))

    backtester = Backtester()

    backtester.backtest_equal_weighted(data.rets.iloc[1000:], use_weights=w_tan)

    print(backtester.single_port_metric(data.prices, backtester.port_val))