import numpy as np
import pandas as pd
from utils.hmm_sampler import SampleHMM


def load_data_get_ret(path='../data/adjusted_close_price_series_load.xlsx'):
    df = pd.read_excel(path, header=2, index_col='Time / Name')
    df.dropna(inplace=True)

    df_ret = df[['Hedge Funds Global', 'MSCI World', 'MSCI Emerging Markets',
                     'Barclays US Treasury Bond Index', 'S&P Listed Private Equity Index',
                     'European Public Real Estate', 'S&P Crude Oil Index', 'Gold']].pct_change()

    df_ret.dropna(inplace=True)

    return df_ret

def get_cov_mat(df_ret):
    return df_ret.cov()

class MPC:
    """
    Solve the convec minimization problem in the MPC approach.

    Parameters
    ----------
    ret_pred : ndarray of shape (n_preds, n_assets)
        Return predictions for each asset h time steps into the future.
    covariances : ndarray of shape (n_assets, n_assets)
        Covarince matrix of returns
    prev_port_ts : ndarray of dynamic length
        Times series of portfolio value at all previous time steps
    """

    def __init__(self, ret_pred, covariances, prev_port_vals, max_drawdown=0.4, gamma_0=5, kappa1=0.004,
                 rho2=0.0005, max_holding=0.4, eps = 0.0000001):

        self.max_drawdown = max_drawdown
        self.gamma_0 = gamma_0
        self.kappa1 = kappa1
        self.rho2 = rho2
        self.max_holding = max_holding
        self.eps = eps

        self.ret_pred = np.array(ret_pred)
        self.cov = np.array(covariances)
        self.prev_port_vals = prev_port_vals

        self.n_assets = self.cov.shape[0]
        self.n_preds = len(self.ret_pred)
        self.start_weights = np.zeros(self.n_assets)
        self.start_weights[-1] = 1.

        # Dummy variables
        #self.weights = np.array([1/self.n_assets]*120).reshape(15,8)

    def port_ret(self, weights):
        port_ret_ = (self.ret_pred * weights).sum(axis=1)

        # Multiply starting portfolio value with the cumulative product of gross returns
        # Gives the portfolio value at each time h in forecasting sequence
        port_vals_ = self.prev_port_vals[-1] * (1 + port_ret_).cumprod()

        self.port_ret_ = port_ret_
        self.port_vals_ = port_vals_

        return port_ret_, port_vals_

    def trading_cost(self, weights):
        # Insert the initial weight first
        # Has to be done this way since the initial weight is not part of the convex opt. problem
        weights = np.insert(weights, 0, self.start_weights, axis=0)
        delta_weight = np.diff(weights, axis=0)
        trading_cost = self.kappa1 * np.abs(delta_weight)

        return np.sum(trading_cost, axis=1)

    def holding_cost(self, weights):
        holding_cost_ = self.rho2 * np.square(weights)
        return np.sum(holding_cost_, axis=1)

    def drawdown_control(self, weights, port_vals_):
        # From all previous total portfolio values get the highest one
        previous_peak = np.max(self.prev_port_vals)

        # Compute running total of highest peak during forecast period
        combined_peaks = np.append(np.array(previous_peak), port_vals_) # vector of candidate running portfolio peaks
        combined_peaks = np.maximum.accumulate(combined_peaks)[1:]


        drawdown_t = 1 - self.prev_port_vals[-1] / previous_peak
        denom = np.max([self.max_drawdown - drawdown_t, self.eps])  # If drawdown limit is breached use a number very close to zero
        gamma = self.gamma_0 * self.max_drawdown / denom

        return gamma

    def port_risk_control(self, weights):  # TODO get rid of loop
        port_var = np.zeros(self.n_preds)
        for t in range(self.n_preds):
            port_var[t] = weights[t].T @ self.cov @ weights[t]

        return port_var




if __name__ == "__main__":
    sampler = SampleHMM(n_states=2, random_state=1)
    X, true_states = sampler.sample(15)

    df_ret = load_data_get_ret()
    df_ret['rf'] = 0.
    cov = get_cov_mat(df_ret).to_numpy()


    ret_pred = df_ret.iloc[-15:].to_numpy()
    weights = np.array([1/9] * 135).reshape(15, 9)
    prev_port_vals = np.array([100, 105,110,100,105,120,130,150,135,160])


    model = MPC(ret_pred, cov, prev_port_vals)
    port_ret, port_vals = model.port_ret(weights)
    trading_cost_ = model.trading_cost(weights)
    holding_cost = model.holding_cost(weights)

    gamma = model.drawdown_control(weights, port_vals)

    port_var = model.port_risk_control(weights)


