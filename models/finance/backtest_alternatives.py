import warnings

import numpy as np
import pandas as pd
import tqdm

from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.finance.mpc_model import MPC
from models.finance.backtest import FinanceHMM, Backtester
from utils.data_prep import load_returns, load_logreturns, load_prices, DataPrep


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
np.seterr(divide='ignore')

class BacktestAlternatives(Backtester):

    def __init__(self):
        super().__init__()

    def backtest_equal_weighted(self, df_rets, rebal_freq='M', port_val=1000, start_weights=None):
        """
       Backtest an equally weighted portfolio, with specified rebalancing frequency.

       Parameters
       ----------
       df_rets : DataFrame of shape (n_samples, n_assets)
           Historical returns for each asset i. Cash must be at the last column position.
       rebal_freq : int, default=20
            Rebalance frequency. Default is 20, i.e monthly.
       port_val : float, default=1000
           Starting portfolio value.
       start_weights : ndarray of shape (n_assets,)
           Current (known) portfolio weights at the start of backtest. Default is 100% allocation to cash.
           Cash must be the last column in df_rets.
       """
        self.port_val = np.array([0, port_val])
        self.n_assets = df_rets.shape[1]
        equal_weights = np.array([1 / self.n_assets] * self.n_assets)  # Vector of shape (n_assets,)

        if start_weights == None:  # Standard init with 100% allocated to cash
            start_weights = np.zeros(self.n_assets)
            start_weights[-1] = 1.
        else:
            start_weights = start_weights

        weights = start_weights
        trade_cost, turnover = [], []

        # Group data into months - average sample size is 20
        # Then for each month loop over the daily returns and update weights
        # The problem is recursive and thus requires looping done this way
        for month_dt, df_group in tqdm.tqdm(df_rets.groupby(pd.Grouper(freq=rebal_freq))):
            # Compute transaction costs for each month. Subtracted from gross ret the first of the month
            delta_weights = equal_weights - weights
            trans_cost = self.transaction_costs(delta_weights)
            weights = equal_weights  # Reset weights
            for day in range(len(df_group)):
                # Calculate gross returns for portfolio and append it
                if day == 0:
                    gross_ret = (1 + df_group.iloc[day]) * (1-trans_cost)
                else:
                    gross_ret = 1 + df_group.iloc[day]

                new_port_val = weights @ gross_ret * self.port_val[-1]
                self.port_val = np.append(self.port_val, new_port_val)

                new_w = gross_ret * weights
                new_w /= new_w.sum()  # Weights sum to 1
                weights = new_w  # Update weights each iteration

            trade_cost.append(trans_cost)
            turnover.append(np.linalg.norm(delta_weights, ord=1) / 2)  # Half L1 norm

        self.port_val = self.port_val[1:]  # Throw away first observation since it is artificially set to zero

        # Annualized average trading ost
        self.trans_cost = np.array(trade_cost)
        self.annual_trans_cost = 12 / len(self.trans_cost) * self.trans_cost.sum()

        # Compute average annualized portfolio turnover
        self.monthly_turnover = np.array(turnover)
        self.annual_turnover = 12 / len(self.monthly_turnover) * self.monthly_turnover.sum()


if __name__ == "__main__":
    data = DataPrep(out_of_sample=True)
    data.rets = data.rets.loc['2007-09-20':]

    backtest = BacktestAlternatives()

    backtest.backtest_equal_weighted(data.rets, rebal_freq='M')
    metrics = backtest.single_port_metric(data.prices, backtest.port_val, compare_assets=True)
    print(metrics)

    backtest.plot_port_val(data.prices, backtest.port_val, start=None, savefig=None)