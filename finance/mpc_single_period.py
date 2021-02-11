import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.optimize as opt

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

        self.ret_pred = np.array(ret_pred)[0,:]
        self.cov = np.array(covariances)
        self.prev_port_vals = prev_port_vals

        self.n_assets = self.cov.shape[0]
        self.n_preds = len(self.ret_pred)
        self.start_weights = np.zeros(self.n_assets)
        self.start_weights[-1] = 1.
        #self.start_weights = cp.Parameter((self.n_assets), value=self.start_weights)

        # Dummy variables
        #self.weights = np.array([1/self.n_assets]*120).reshape(15,8)

    def port_ret(self, weights):
        port_ret_ = cp.multiply(self.ret_pred, weights)
        return cp.sum(port_ret_)

    def trading_cost(self, weights):
        # Insert the initial weight first
        # Has to be done this way since the initial weight is not part of the convex opt. problem
        delta_weight = self.start_weights - weights
        delta_weight = cp.abs(delta_weight)
        trading_cost = self.kappa1 * delta_weight

        return cp.sum(trading_cost)

    def holding_cost(self, weights):
        holding_cost_ = self.rho2 * cp.square(weights)
        return cp.sum(holding_cost_)

    def drawdown_control(self):
        # From all previous total portfolio values get the highest one
        previous_peak = np.max(self.prev_port_vals)

        drawdown_t = 1 - self.prev_port_vals[-1] / previous_peak
        denom = np.max([self.max_drawdown - drawdown_t, self.eps])  # If drawdown limit is breached use a number very close to zero
        gamma = self.gamma_0 * self.max_drawdown / denom

        return gamma

    def port_risk_control(self, weights):
        port_var = cp.quad_form(weights, self.cov)
        return port_var

    def constraints(self, weights):
        return [cp.sum(weights) == 1]

    def objective_func(self, weights):
        port_ret = self.port_ret(weights)
        trading_cost_ = self.trading_cost(weights)
        holding_cost_ = self.holding_cost(weights)
        gamma = self.drawdown_control()
        port_risk = self.port_risk_control(weights)

        objctive = port_ret - trading_cost_ - holding_cost_ - gamma * port_risk
        return objctive

    def cvxpy_solver(self):
        weights = cp.Variable(self.n_assets)
        print("dimensions of X:", weights.shape)
        print("size of X:", weights.size)
        print("number of dimensions:", weights.ndim)
        print("dimensions of sum(X):", cp.sum(weights).shape)

        objective = cp.Maximize(self.objective_func(weights))
        constraints = self.constraints(weights)
        prob = cp.Problem(objective, constraints)

        print('Optimal value: ', prob.solve(verbose=True))
        print("Optimal var")
        print(weights.value)
        print(np.sum(weights.value))


if __name__ == "__main__":
    sampler = SampleHMM(n_states=2, random_state=1)
    X, true_states = sampler.sample(15)

    df_ret = load_data_get_ret()
    df_ret['rf'] = 0.
    cov = get_cov_mat(df_ret).to_numpy()

    ret_pred = df_ret.iloc[-1:].to_numpy()
    #weights = np.array([1/9] * 135).reshape(15, 9)
    prev_port_vals = np.array([100, 105,110,100,105,120,130,150,135,160])

    model = MPC(ret_pred, cov, prev_port_vals)

    model.cvxpy_solver()


