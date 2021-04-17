import numpy as np; np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import tqdm

from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.finance.mpc_model import MPC
from utils.data_prep import load_returns , load_logreturns, load_prices

import warnings
warnings.filterwarnings('ignore')

""" TODO
In Backtester.get_asset_dist when only 1 state is present the other takes on mean
values of 0 and covariances of 0. Is this a good result ?

Sometimes a state is not vistited or only has 1 observation in which case covariance matrix will return null results.
We need a solution for this; currently setting it equal to zero in those cases.

Create some way to do in-sample crossvalidation for hyperparameters

TRANSACTION COSTS: Do we subtract transaction_costs() method fom gross returns or multiply by (1-trans_costs)

SHRINKAGE: Sort portfolios according to variance
    EVMA OR Hyperbolic
    Eksponentielt vægtet
    

"""

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
    def __init__(self, window_len=1500):
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
                     start_weights=None, max_drawdown=0.4, max_leverage=2.0, gamma_0=5, kappa1=0.008,
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
           Current portfolio value.
       start_weights : ndarray of shape (n_assets,)
           Current (known) portfolio weights at the start of backtest. Default is 100% allocation to cash.
           Cash must be the last column in df_rets.
       """
        self.port_val = np.array([0, port_val])
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
        trade_cost = []

        # Instantiate MPC object
        mpc_solver = MPC(rets=preds[0], covariances=covariances[0], prev_port_vals=self.port_val,
                         start_weights=self.weights[0], max_drawdown=max_drawdown, gamma_0=gamma_0,
                         kappa1=kappa1, rho2=rho2, max_holding=max_holding, max_leverage=max_leverage,
                         short_cons=short_cons, eps=eps)

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
            new_port_val = (gross_ret-shorting_cost) * (1-trans_cost) * self.port_val[-1]  # TODO trans+short costs???
            self.port_val = np.append(self.port_val, new_port_val)

            trade_cost.append(trans_cost)

        self.port_val = self.port_val[1:]  # Throw away first observation since it is artificially set to zero
        self.gamma = gamma
        self.trans_cost = np.array(trade_cost)

        return self.weights, self.port_val, gamma

    def short_costs(self, weights, rf_return):
        """
        Compute shorting costs, assuming a fee equal to the risk-free asset is paid.
        """
        short_weights = weights[:-1][weights[:-1] < 0.0].sum()  # Sum of all port weights below 0.0
        return -short_weights * rf_return

    def transaction_costs(self, delta_weights, trans_cost=0.001):
        """
        Compute transaction costs. Assumes no costs in risk-free asset and equal cost to
        buying and selling assets.
        """
        delta_weights = delta_weights[:-1]  # Remove risk-free asset as it doesn't have trading costs
        delta_weights = np.abs(delta_weights).sum()  # abs since same price for buying/selling

        return delta_weights * trans_cost

    def performance_metrics(self, df, port_val, compare_assets=False):
        """Compute performance metrics for a given portfolio/asset"""
        # Merge port_val with data
        df = df.iloc[-len(port_val):]
        df['port_val'] = port_val
        df.dropna(inplace=True)
        df_ret = df.pct_change().dropna()

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

        if compare_assets == True:
            ret = df_ret.drop('T-bills rf', axis=1)
            cagr = ((1 + ret).prod(axis=0)) ** (1 / n_years) - 1
            std = ret.std(axis=0, ddof=1) * np.sqrt(252)

            excess_ret = df_ret.subtract(df_ret['T-bills rf'], axis=0).drop('T-bills rf', axis=1)
            excess_cagr = ((1 + excess_ret).prod(axis=0)) ** (1 / n_years) - 1
            excess_std = excess_ret.std(axis=0 ,ddof=1) * np.sqrt(252)
            sharpe = excess_cagr / excess_std

            df = df.drop('T-bills rf', axis=1)
            peaks = df.cummax(axis=0)
            drawdown = -(df - peaks) / peaks
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
        else:
            metrics = {'excess_return': excess_cagr,
                       'std': excess_std,
                       'sharpe': sharpe,
                       'max_drawdown': max_drawdown,
                       'max_drawdown_dur': drawdown_dur,
                       'calmar_ratio': calmar}

        return metrics


if __name__ == "__main__":
    path = '../../analysis/portfolio_exercise/output_data/'
    df_logret = load_logreturns()
    X = df_logret["MSCI World"]

    model1 = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)
    backtester = Backtester()

    #preds, cov = backtester.rolling_preds_cov_from_hmm(X, df_logret, model1, window_len=1700, shrinkage_factor=(0.3, 0.3), verbose=True)
    #np.save(path + 'rolling_preds.npy', preds)
    #np.save(path + 'rolling_cov.npy', cov)

    df_ret = load_returns()
    preds = np.load(path + 'rolling_preds.npy')
    cov = np.load(path + 'rolling_cov.npy')

    #weights, port_val, gamma = backtester.backtest_mpc(df_ret, preds, cov, short_cons='LLO')

    #np.save(path + 'mpc_weights.npy', weights)
    #np.save(path + 'port_val.npy', port_val)
    #np.save(path + 'gamma.npy', gamma)

    port_val = np.load(path + 'port_val.npy')
    weights = np.load(path + 'mpc_weights.npy')
    df = load_prices()

    metrics = backtester.performance_metrics(df, port_val, compare_assets=True)

    print(metrics)
    #print('transaction costs:', (1-backtester.trans_cost).prod())
    #print('highest trans cost', backtester.trans_cost.max())

    df = df.iloc[-len(port_val):]