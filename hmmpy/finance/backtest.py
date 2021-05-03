""" TODO
In Backtester.get_asset_dist when only 1 state is present the other takes on mean
values of 0 and covariances of 0. Is this a good result ?

Sometimes a state is not vistited or only has 1 observation in which case covariance matrix will return null results.
We need a solution for this; currently setting it equal to zero in those cases.

Create some way to do in-sample crossvalidation for hyperparameters

SHRINKAGE: Sort portfolios according to variance
    EVMA OR Hyperbolic
    Eksponentielt vægtet


"""
import warnings

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

from hmmpy.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from hmmpy.finance.mpc_model import MPC
from hmmpy.utils.data_prep import DataPrep

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
np.seterr(divide='ignore')

class FinanceHMM:
    """
    Class to compute multivariate mixture distributions from n_assets based on a given HMM.

    Computes posteriors, state sequences as well as expected and forecasted returns and standard deviations.
    Transforms lognormal multivariate distributions into normal distributions and combines them into mixtures.

    Parameters
    ----------
    X : ndarray of shape (n_samples,)
        Times series data used to train the HMM.
    df : DataFrame of shape (n_samples, n_assets)
        Times series data used when estimating expected returns and covariances.
    model : hidden markov model
        Hidden Markov Model object.

    Attributes
    ----------
    preds : ndarray of shape (n_samples-window_len, n_preds, n_assets)
        mean predictions for each asset h time steps into the future at each time t.
    cov : ndarray of shape(n_samples-window_len, n_preds, n_assets, n_assets)
        predicted covariance matrix h time steps into the future at each time t.
    """

    def __init__(self, model):
        self.model = model
        self.n_states = model.n_states
        self.n_assets = None

    def get_cond_asset_dist(self, df, state_sequence):
        """
        Compute conditional multivariate normal distribution of all assets in each state.

        Assumes returns follow a multivariate log-normal distribution. Proceeds by first
        getting the conditional log of means and covariances and then transforming them
        back into normal varibles.

        Parameters
        ----------
        df : DataFrame of shape (n_samples, n_assets)
            log-returns for assets
        state_sequence : ndarray of shape (n_samples,)
            Decoded state sequence

        Returns
        -------
        mu : ndarray of shape (n_states, n_assets)
            Conditional mean value of each assets
        cov : ndarray of shape (n_states, n_assets, n_assets)
            Conditional covariance matrix
        """
        self.n_assets = df.shape[1]
        df = df.iloc[-len(state_sequence):]
        df['state_sequence'] = state_sequence
        groupby_state = df.groupby('state_sequence')
        log_mu, log_cov = groupby_state.mean(), groupby_state.cov()
        state_count = groupby_state.count().max(axis=1)  # Num obs in each state

        mu = np.zeros(shape=(self.n_states, self.n_assets))
        cov = np.zeros(shape=(self.n_states, self.n_assets, self.n_assets))

        # Loop through n_states present in current sample
        for s in log_mu.index:
            if state_count[s] > 1:  # If state_count not >1, covariance will return NaN
                mu[s], cov[s] = self.logcov_to_cov(log_mu.loc[s], log_cov.loc[s])

        return mu, cov

    def get_uncond_asset_dist(self, posteriors, cond_mu, cond_cov):
        """
        Compute unconditional multivariate normal distribution of all assets.

        Parameters
        ----------
        posteriors: ndarray of shape (n_preds, n_states)
            predicted posterior probability of being in state i at time t+h.
        cond_mu : ndarray of shape (n_states, n_assets)
            Conditional mean value of each assets
        cond_cov : ndarray of shape (n_states, n_assets, n_assets)
            Conditional covariance matrix
        Returns
        -------
        pred_mu : ndarray of shape (n_preds, n_assets)
            Conditional mean value of each assets
        pred_cov : ndarray of shape (n_preds, n_assets, n_assets)
            Conditional covariance matrix
        """
        pred_mu = np.inner(cond_mu.T, posteriors).T  # shape (n_preds, n_assets)

        cov_x1 = np.inner(posteriors, cond_cov.T)  # shape (n_preds, n_assets, n_assets)
        cov_x2 = pred_mu - cond_mu[:, np.newaxis]  # shape (n_states, n_preds)
        cov_x3 = np.einsum('ijk,ijk->ij', cov_x2, cov_x2)  # Equal to np.sum(X**2, axis=-1)
        cov_x4 = np.einsum('ij,ij->i', cov_x3.T, posteriors)  # Equal to np.sum(X3*posteriors, axis=1)
        pred_cov = cov_x1 + cov_x4[:, np.newaxis, np.newaxis]  # shape (n_preds, n_assets, n_assets)

        return pred_mu, pred_cov

    @staticmethod
    def logcov_to_cov(log_mu, log_cov):
        """
        Transforms log returns' means and covariances back into regular formats.

        Parameters
        ----------
        log_mu : DataFrame of shape (n_assets,)
        log_cov : DataFrame of shape (n_assets, n_assets)

        Returns
        -------
        mu : ndarray of shape (n_assets)
            Mean value of each assets
        cov : ndarray of shape (n_assets, n_assets)
            Covariance matrix
        """
        diag = np.diag(log_cov)
        mu = np.exp(log_mu + np.diag(log_cov) / 2) - 1
        x1 = np.outer(mu, mu)  # Multiply all combinations of the vector mu -> 2-D array
        x2 = np.outer(diag, diag) / 2
        cov = np.exp(x1 + x2) * (np.exp(log_cov) - 1)

        return mu, cov

    def stein_shrinkage(self, cond_cov, shrinkage_factor=(0.2, 0.4)):
        """Stein-type shrinkage of conditional covariance matrices"""
        shrinkage_factor = np.array(shrinkage_factor)

        # Turn it into 3D to make it broadcastable with cond_cov
        shrink_3d = shrinkage_factor[:, np.newaxis, np.newaxis]
        term1 = (1-shrink_3d) * cond_cov

        # Turn term2 into 3D to make it broadcastable with term3
        term2 = (shrinkage_factor * np.trace(cond_cov.T) * 1/self.n_assets)  # Shape (n_states,)
        term3 = np.broadcast_to(np.identity(self.n_assets)[..., np.newaxis],
                                (self.n_assets,self.n_assets,self.n_states)).T  # Shape (n_states, n_assets, n_assets)
        term4 = term2[:, np.newaxis, np.newaxis] * term3
        cond_cov = term1 + term4
        return cond_cov

    def fit_model_get_uncond_dist(self, X, df, n_preds=15, shrinkage_factor=(0.2, 0.4), verbose=False):
        """
        From data, fit hmm model, predict posteriors probabilities and return unconditional distribution.

        Wraps model.fit_predict, get_cond_asset_dist and get_uncond_asset_dist methods into one.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Time series of data
        df : DataFrame of shape (n_samples, n_assets)
            Historical returns for each asset i.
        n_preds : int, default=15
            Number of h predictions
        verbose : boolean, default=False
            Get verbose output

        Returns
        -------
        pred_mu : ndarray of shape (n_preds, n_assets)
            Conditional mean value of each assets
        pred_cov : ndarray of shape (n_preds, n_assets, n_assets)
            Conditional covariance matrix
        """
        self.n_assets = df.shape[1]
        # fit model, return decoded historical state sequnce and n predictions
        # state_sequence is 1D-array with same length as X_rolling
        # posteriors is 2D-array with shape (n_preds, n_states)
        state_sequence, posteriors = self.model.fit_predict(X, n_preds=n_preds, verbose=verbose)

        # Compute conditional mixture distributions in rolling period
        cond_mu, cond_cov = \
            self.get_cond_asset_dist(df, state_sequence)  # shapes (n_states, n_assets), (n_states, n_assets, n_assets)
        cond_cov = self.stein_shrinkage(cond_cov, shrinkage_factor=shrinkage_factor)

        # Transform into unconditional moments at time t
        # Combine with posteriors to also predict moments h steps into future
        # shapes (n_preds, n_assets), (n_preds, n_assets, n_assets)
        pred_mu, pred_cov = self.get_uncond_asset_dist(posteriors, cond_mu, cond_cov)

        return pred_mu, pred_cov, posteriors, state_sequence

class Backtester:
    """
    Backtester for Hidden Markov Models.

    Parameters
    ----------

    Attributes
    ----------
    preds : ndarray of shape (n_samples-window_len, n_preds, n_assets)
        mean predictions for each asset h time steps into the future at each time t.
    cov : ndarray of shape(n_samples-window_len, n_preds, n_assets, n_assets)
        predicted covariance matrix h time steps into the future at each time t.
    """
    def __init__(self, window_len=1700):
        self.preds = None
        self.cov = None
        self.n_states = None
        self.n_assets = None
        self.window_len = window_len

    def rolling_preds_cov_from_hmm(self, X, df_logret, model, n_preds=15, window_len=None, shrinkage_factor=(0.3, 0.3), verbose=False):
        """
        Backtest based on rolling windows.

        Fits a Hidden Markov model within each rolling window and computes the unconditional
        multivariate normal mixture distributions for each asset in the defined universe.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
        Log-returns. Times series data used to train the HMM.
        df_logret : DataFrame of shape (n_samples, n_assets)
            Log-returns. Times series data used when estimating expected returns and covariances.
        model : hidden markov model
            Hidden Markov Model object
        n_preds : int, default=15
            Number of h predictions
        window_len : int, default=1500
        verbose : boolean, default=False
            Make output verbose

        Returns
        -------
        preds : ndarray of shape (n_samples-window_len, n_preds, n_assets)
            Unconditional mean values for each asset
        cov : ndarray of shape (n_samples-window_len, n_preds, n_assets, n_assets)
            Unconditional covariance matrix at each time step t, h steps into future
        """
        self.n_states = model.n_states
        self.n_assets = df_logret.shape[1]

        if window_len == None:  # Ensure class and function window_lens match
            window_len = self.window_len
        else:
            self.window_len = window_len

        finance_hmm = FinanceHMM(model)  # class for computing asset distributions and predictions.

        # Create 3- and 4-D array to store predictions and covariances
        self.preds = np.empty(shape=(len(df_logret) - window_len, n_preds, self.n_assets))  # 3-D array
        self.cov = np.empty(shape=(len(df_logret) - window_len, n_preds, self.n_assets, self.n_assets))  # 4-D array
        self.timestamp = np.empty(shape=len(df_logret) - window_len, dtype=object)

        for t in tqdm.trange(window_len, len(df_logret)):
            # Slice data into rolling sequences
            df_rolling = df_logret.iloc[t-window_len: t]
            X_rolling = X.iloc[t-window_len: t]

            # fit rolling data with model, return predicted means and covariances, posteriors and state sequence
            pred_mu, pred_cov, posteriors, state_sequence = \
                finance_hmm.fit_model_get_uncond_dist(
                    X_rolling, df_rolling, shrinkage_factor=shrinkage_factor, n_preds=n_preds, verbose=verbose)

            self.timestamp[t - window_len] = df_rolling.index[-1]
            self.preds[t - window_len] = pred_mu
            self.cov[t - window_len] = pred_cov

        return self.preds, self.cov

    def backtest_mpc(self, df_rets, preds, covariances, n_preds=15, port_val=1000,
                     start_weights=None, max_drawdown=0.4, max_holding_rf=1.,
                     max_leverage=2.0, gamma_0=5, kappa1=0.008,
                     rho2=0.0005, max_holding=0.4, short_cons="LLO", eps=1e-6):
        """
       Wrapper for backtesting MPC models on given data and predictions.

       Parameters
       ----------
       df_rets : DataFrame of shape (n_samples, n_assets)
           Historical returns for each asset i. Cash must be at the last column position.
       preds : ndarray of shape (n_samples, n_preds, n_assets)
           list of return predictions for each asset h time steps into the future. Each element in list contains,
           from time t, predictions h time steps into the future.
       covariances : ndarray of shape (n_samples, n_preds, n_assets, n_assets)
           list of covariance matrix of returns for each time step t.
       port_val : float, default=1000
           Starting portfolio value.
       start_weights : ndarray of shape (n_assets,)
           Current (known) portfolio weights at the start of backtest. Default is 100% allocation to cash.
           Cash must be the last column in df_rets.
       """
        self.port_val = np.array([0, port_val])
        self.port_ret = np.array([1, 1])
        self.n_assets = df_rets.shape[1]
        self.n_preds = n_preds

        df_rets = df_rets.iloc[-len(preds):]  # Slice returns to match preds

        if start_weights == None:  # Standard init with 100% allocated to cash
            start_weights = np.zeros(self.n_assets)
            start_weights[-1] = 1.
        else:
            start_weights = start_weights

        self.weights = np.zeros(shape=(len(preds) + 1, self.n_assets))  # len(preds) + 1 to include start weights
        self.weights[0] = start_weights

        gamma = np.array([])  # empty array
        trade_cost, turnover = [], []

        # Instantiate MPC object
        mpc_solver = MPC(rets=preds[0], covariances=covariances[0], prev_port_vals=self.port_val,
                         start_weights=self.weights[0], max_drawdown=max_drawdown, gamma_0=gamma_0,
                         kappa1=kappa1, rho2=rho2, max_holding=max_holding, max_holding_rf=max_holding_rf
                         ,max_leverage=max_leverage, short_cons=short_cons, eps=eps)

        for t in tqdm.trange(preds.shape[0]):
            # Update MPC object
            mpc_solver.rets = np.array(preds[t])
            mpc_solver.cov = np.array(covariances[t])
            mpc_solver.start_weights = self.weights[t]
            mpc_solver.prev_port_vals = self.port_val

            # Solve MPC problem at time t and save weights
            weights_mpc = mpc_solver.cvxpy_solver(verbose=False)  # ndarray of shape (n_preds, n_assets)
            self.weights[t + 1] = weights_mpc[0]  # Only use first forecasted weights
            gamma = np.append(gamma, mpc_solver.gamma)
            delta_weights = self.weights[t] - self.weights[t-1]

            # self.weights and df_rets are one shifted to each other. Time periods should match.
            gross_ret = (self.weights[t + 1] @ (1 + df_rets.iloc[t]))
            shorting_cost = self.short_costs(self.weights[t + 1], rf_return=df_rets.iloc[t, -1])
            trans_cost = self.transaction_costs(delta_weights, trans_cost=0.001)
            port_ret = (gross_ret-shorting_cost) * (1-trans_cost)
            new_port_val = port_ret * self.port_val[-1]
            self.port_ret = np.append(self.port_ret, port_ret)
            self.port_val = np.append(self.port_val, new_port_val)

            trade_cost.append(trans_cost)
            turnover.append(np.linalg.norm(delta_weights, ord=1) / 2)  # Half L1 norm

        self.port_val = self.port_val[1:]  # Throw away first observation since it is artificially set to zero
        self.port_ret = self.port_ret[2:]
        self.gamma = gamma

        # Annualized average trading ost
        self.trans_cost = np.array(trade_cost)
        self.annual_trans_cost = 252 / len(self.trans_cost) * self.trans_cost.sum()

        # Compute average annualized portfolio turnover
        self.daily_turnover = np.array(turnover)
        self.annual_turnover = 252 / len(self.daily_turnover) * self.daily_turnover.sum()

        # Compute return & std.
        n_years = len(self.port_val) / 252
        annual_ret = self.port_ret.prod()**(1/n_years) - 1
        annual_std = self.port_ret.std(ddof=1) * np.sqrt(252)

        return annual_ret, annual_std, self.annual_turnover

    def gridsearch_mpc(self, grid, df_rets, preds, covariances, n_preds=15, port_val=1000,
                     start_weights=None, max_drawdown=0.4, max_leverage=2.0, gamma_0=5, kappa1=0.008,
                     rho2=0.0005, max_holding=0.4, short_cons="LO", eps=1e-6):
        results = pd.DataFrame()
        for max_holding in grid['max_holding']:
            for trans_costs in grid['trans_costs']:
                for holding_costs in grid['holding_costs']:
                    print(f"""Computing grid -- max_holding {max_holding} -- trans_costs {trans_costs} holding_costs {holding_costs}""")
                    try:
                        annual_ret, annual_std, annual_turnover = \
                            self.backtest_mpc(df_rets, preds, covariances, n_preds=n_preds, port_val=port_val,
                         start_weights=start_weights, max_drawdown=max_drawdown, max_leverage=max_leverage,
                        gamma_0=gamma_0, kappa1=trans_costs, rho2=holding_costs, max_holding=max_holding,
                                          short_cons=short_cons, eps=eps)

                        results_dict = {'max_holding': max_holding,
                                        'trans_costs': trans_costs,
                                        'holding_costs': holding_costs,
                                        'return': annual_ret,
                                        'std': annual_std,
                                        'turnover': annual_turnover}
                        print(results_dict)
                        results = results.append(results_dict, ignore_index=True)
                    except:
                        print('No convergence')
                        continue
        self.gridsearch_df = results
        return results

    def mpc_gammas_shortcons(self, gammas, constraints,
                             data, preds, covariances, n_preds=15, port_val=1000,
                             start_weights=None, max_holding_rf=1.,
                             max_leverage=2.0, trans_costs=0.001,
                             holding_costs=0.0000, max_holding=0.2, eps=1e-6):

        df = pd.DataFrame()


        for constr in constraints:
            print(f'Backtesting for params {constr}')
            results = {f'gamma_{i}': [] for i in gammas}
            short_con = constr[0]
            max_drawdown = constr[1]
            for gamma in gammas:
                self.backtest_mpc(data.rets, preds, covariances, n_preds=n_preds, port_val=port_val,
                                  start_weights=start_weights, max_drawdown=max_drawdown, max_leverage=max_leverage,
                                  gamma_0=gamma, kappa1=trans_costs, rho2=holding_costs, max_holding=max_holding,
                                  short_cons=short_con, eps=eps)

                results[f'gamma_{gamma}'] = self.port_val

            df_temp = pd.DataFrame(results)
            df_temp['short_cons'] = short_con
            df_temp['D_max'] = max_drawdown
            df_temp['timestamp'] = data.rets.index[-len(df_temp):]
            df_temp['T-bills rf'] = data.prices['T-bills rf'].iloc[-len(df_temp):].values
            df = df.append(df_temp)

        # self.annual_turnover, self.annual_trans_cost, self.port_val
        self.port_val_df = df
        return df


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

    def short_costs(self, weights, rf_return):
        """
        Compute shorting costs, assuming a fee equal to the risk-free asset is paid.
        """
        weights_no_rf = weights[:-1]  # Remove risk-free asset from array
        short_weights = weights_no_rf[weights_no_rf < 0.0].sum()  # Sum of all port weights below 0.0
        return -short_weights * rf_return

    def transaction_costs(self, delta_weights, trans_cost=0.001):
        """
        Compute transaction costs. Assumes no costs in risk-free asset and equal cost to
        buying and selling assets.
        """
        delta_weights = delta_weights[:-1]  # Remove risk-free asset as it doesn't have trading costs
        delta_weights = np.abs(delta_weights).sum()  # abs since same price for buying/selling

        return delta_weights * trans_cost

    def asset_metrics(self, df_prices):
        """Compute performance metrics for a given portfolio/asset"""
        df_ret = df_prices.pct_change().dropna()
        n_years = len(df_ret) / 252

        # Get regular cagr and std
        ret = df_ret.drop('T-bills rf', axis=1)
        cagr = ((1 + ret).prod(axis=0)) ** (1 / n_years) - 1
        std = ret.std(axis=0, ddof=1) * np.sqrt(252)

        # Compute metrics in excess of the risk-free asset
        excess_ret = df_ret.subtract(df_ret['T-bills rf'], axis=0).drop('T-bills rf', axis=1)
        excess_cagr = ((1 + excess_ret).prod(axis=0)) ** (1 / n_years) - 1
        excess_std = excess_ret.std(axis=0 ,ddof=1) * np.sqrt(252)
        sharpe = excess_cagr / excess_std

        df_prices = df_prices.drop('T-bills rf', axis=1)
        peaks = df_prices.cummax(axis=0)
        drawdown = -(df_prices - peaks) / peaks
        max_drawdown = drawdown.max(axis=0)
        calmar = excess_cagr / max_drawdown

        metrics = {'return': cagr,
                   'std': std,
                   'excess_return': excess_cagr,
                   'excess_std': excess_std,
                   'sharpe': sharpe,
                   'max_drawdown': max_drawdown,
                   'calmar_ratio': calmar}

        metrics = pd.DataFrame(metrics)

        return metrics

    def single_port_metric(self, df_prices, port_val, compare_assets=False):
        """Compute performance metrics for a given portfolio/asset"""
        # Merge port_val with data
        df_prices = df_prices.iloc[-len(port_val):]
        df_prices['port_val'] = port_val
        df_prices.dropna(inplace=True)
        df_ret = df_prices.pct_change().dropna()

        # Annual returns, std
        n_years = len(port_val) / 252
        excess_ret = df_ret['port_val'] - df_ret['T-bills rf']

        excess_cagr = ((1+excess_ret).prod())**(1/n_years) - 1
        excess_std = excess_ret.std(ddof=1) * np.sqrt(252)
        sharpe = excess_cagr / excess_std

        # Drawdown
        peaks = np.maximum.accumulate(port_val)
        drawdown = -(port_val-peaks) / peaks
        max_drawdown = np.max(drawdown)
        max_drawdown_end = np.argmax(drawdown)
        max_drawdown_beg = np.argmax(port_val[:max_drawdown_end])
        drawdown_dur = max_drawdown_end - max_drawdown_beg  # TODO not showing correct values
        calmar = excess_cagr / max_drawdown

        metrics = {'excess_return': excess_cagr,
                   'excess_std': excess_std,
                   'sharpe': sharpe,
                   'max_drawdown': max_drawdown,
                   'max_drawdown_dur': drawdown_dur,
                   'calmar_ratio': calmar}

        return metrics

    def mulitple_port_metrics(self, df_port_val):
        """Compute performance metrics for a given portfolio/asset"""
        # Merge port_val with data
        """
        df_prices = df_prices.iloc[-len(port_val):]
        df_prices = df_prices[['T-bills rf']]  # Remove other assets
        df_prices['port_val'] = port_val
        df_prices.dropna(inplace=True)
        df_ret = df_prices.pct_change().dropna()
        """
        df = pd.DataFrame()
        for type, df_groupby in df_port_val.groupby(['short_cons', 'D_max']):
            df_groupby['T-bills rf']
            df_prices = df_groupby.drop(columns=['short_cons', 'D_max', 'timestamp'])
            df_rets = df_prices.pct_change().dropna()

            # Annual returns, std
            n_years = len(df_rets) / 252
            ret = df_rets.drop('T-bills rf', axis=1)
            cagr = ((1 + ret).prod(axis=0)) ** (1 / n_years) - 1
            std = ret.std(axis=0, ddof=1) * np.sqrt(252)

            excess_ret = df_rets.subtract(df_rets['T-bills rf'], axis=0).drop('T-bills rf', axis=1)
            excess_cagr = ((1 + excess_ret).prod(axis=0)) ** (1 / n_years) - 1
            excess_std = excess_ret.std(axis=0 ,ddof=1) * np.sqrt(252)
            sharpe = excess_cagr / excess_std

            df_prices = df_prices.drop('T-bills rf', axis=1)
            peaks = df_prices.cummax(axis=0)
            drawdown = -(df_prices - peaks) / peaks
            max_drawdown = drawdown.max(axis=0)
            """
            max_drawdown_end = np.argmax(drawdown, axis=0)
            max_drawdown_beg = np.argmax(drawdown[:max_drawdown_end], axis=0)
            drawdown_dur = max_drawdown_end - max_drawdown_beg  # TODO not showing correct values
            """
            calmar = excess_cagr / max_drawdown

            metrics = {'return': cagr,
                       'std': std,
                       'excess_return': excess_cagr,
                       'excess_std': excess_std,
                       'sharpe': sharpe,
                       'max_drawdown': max_drawdown,
                       'calmar_ratio': calmar}

            df_temp = pd.DataFrame(metrics)
            df_temp['short_cons'] = type[0]
            df_temp['D_max'] = type[1]
            df = df.append(df_temp)

        return df

    def plot_port_val(self, data, mpc_val, equal_w_val, start=None, savefig=None):
        # Prepare data
        equal_w_val = equal_w_val[-len(mpc_val):]
        data.dropna(inplace=True)
        data = data.iloc[-len(mpc_val):]
        data['MPC'] = mpc_val
        data['1/n'] = equal_w_val
        data = data[['MPC', '1/n']]  # Drop all other cols

        if not start == None:
            data = data.loc[start:]

        data = data / data.iloc[0] * 100

        # Plotting
        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(15,10))

        ax.plot(data.index, data)
        # ax[0].set_yscale('log')
        ax.set_ylabel('$P_t$')

        plt.tight_layout()

        if not savefig == None:
            plt.savefig('./images/' + savefig)
        plt.show()

if __name__ == "__main__":
    data = DataPrep(out_of_sample=True)
    df_port_val = data.prices
    df_ret = data.rets
    df_logret = data.logrets
    X = df_logret["S&P 500 "]

    model1 = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)
    backtester = Backtester()

    #preds = np.load(path + 'rolling_preds.npy')
    #cov = np.load(path + 'rolling_cov.npy')
    #weights, port_val, gamma = backtester.backtest_mpc(df_ret, preds, cov, short_cons='LLO')
    #port_val = np.load(path + 'port_val.npy')
    #weights = np.load(path + 'mpc_weights.npy')
    #metrics = backtester.asset_metrics(df_prices)

