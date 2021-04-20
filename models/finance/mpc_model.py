import numpy as np
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import matplotlib.pyplot as plt
import cvxpy as cp

from utils.data_prep import load_returns

""" TODO

"""

class MPC:
    """
    Solve the model predictive control problem using convex optimization.

    This class is strictly made for use with the CVXPY library. All methods are created to be
    compatible with this library and as such use of Numpy or Pandas is limited.

    Parameters
    ----------
    rets : ndarray of shape (n_preds, n_assets)
        Return predictions for each asset h time steps into the future.
    covariances : ndarray of shape (n_assets, n_assets)
        Covarince matrix of returns
    prev_port_vals : ndarray of dynamic length
        Times series of portfolio value at all previous time steps
    start_weights : ndarray of shape (n_assets,)
        Current (known) portfolio weights at time t.
    short_cons : str, default='LLO'
        How portfolio shorting is determined.
        'LLO': Leveraged-Long-Only, allowed to short risk-free asset.
        'long_only': No shorting allowed.
        'max_leverage': Shorting constraint entirely determined by max_leverage parameter
    max_leverage : float, default=2.0
        Maximum tolerable leverage defined as the sum of absolute values of the
        decimal weight in each asset apart from the risk-free asset.
    """

    def __init__(self, rets, covariances, prev_port_vals, start_weights,
                 max_drawdown=0.4, gamma_0=5, kappa1=0.004, kappa2=0.,
                 rho1=0., rho2=0.0005, max_holding=0.4, max_leverage=2.0,
                 short_cons='LLO', eps=0.0000001):
        if not short_cons in ['LLO', 'leveraged_long_only', 'LO', 'long_only', 'max_leverage', None]:
            raise Exception("""short_cons must be declared as a str.
                            Options include 'LLO', 'leveraged_long_only', 'LO', 'long_only', 'max_leverage', None.""")

        self.max_drawdown = max_drawdown
        self.gamma_0 = gamma_0
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.rho1 = rho1
        self.rho2 = rho2
        self.max_holding = max_holding
        self.max_leverage = max_leverage
        self.short_cons = short_cons
        self.eps = eps

        self.rets = np.array(rets)
        self.cov = np.array(covariances)
        self.prev_port_vals = prev_port_vals

        self.n_assets = self.rets.shape[1]
        self.n_preds = len(self.rets)
        self.start_weights = start_weights

    def single_period_objective_func(self, current_weights, prev_weights, rets, cov):
        """
        Compiles all individual components in the objective function into one.

        The method works for a single period. Remaining periods are bundled in cvxpy_solver method.

        Parameters
        ----------
        current_weights : CVXPY Variable of shape (n_assets,)
            Portfolio weights in the given time step t
        prev_weights : CVXPY Variable of shape (n_assets,)
            Portfolio weights in the previous time step t-1
        rets : ndarray of shape (n_assets,)
            Predicted returns for each asset at time t

        Returns
        -------
        objective : float
            Scalar value of objective function evaluated at time t.
        """
        port_ret = self._port_ret(current_weights, rets)
        trading_cost_ = self._trading_cost(current_weights, prev_weights)
        holding_cost_ = self._holding_cost(current_weights)
        port_risk = self._port_risk_control(current_weights, cov)

        objctive = port_ret - trading_cost_ - holding_cost_ - self.gamma * port_risk
        return objctive

    def single_period_constraints(self, current_weights):
        """
        Construct portfolio constraints for a single period. Bundled across time in cvxpy_solver method.

        Parameters
        ----------
        current_weights : CVXPY Variable of shape (n_assets,)
            Portfolio weights in the given time step t

        Returns
        -------
        constraints : list
            list containing all construct for time period t.
        """
        constraints = [cp.sum(current_weights) == 1, cp.abs(current_weights) <= self.max_holding]

        # Leverage constraint. Exclude risk-free asset from this computation
        constraints += [cp.sum(cp.abs(current_weights[:-1])) <= self.max_leverage]
        if self.short_cons in ['LLO', 'leveraged_long_only']:  # Can only short risk-free asset
            constraints += [current_weights[:-1] >= 0]
        elif self.short_cons in ['LO', 'long_only']:
            constraints += [current_weights >= 0]
        elif self.short_cons in ['max_leverage', None]:
            pass

        return constraints

    def cvxpy_solver(self, verbose=False):
        """
        Compile the optimization problem into CVXPY objects and solve.

        Returns optimal weights for each forecasted time period as ndarray of shape (n_preds, n_assets).
        """
        # variable with shape h+1 predictions so first row
        # can be the known (non-variable) portfolio weight at time t
        self.gamma = self._gamma_from_drawdown_control()

        weights = cp.Variable(shape=(self.n_preds + 1, self.n_assets))
        objective = 0
        constr = []

        # Loop through each row in the weights variable and construct the optimization problem
        # Note this loop is very cpu-light since no actual computations takes place inside it
        for t in range(1, weights.shape[0]):
            # sum problem objectives. Weights are shifted 1 period forward compared to self.rets
            # Concatenates objective and constraints in lists
            objective += self.single_period_objective_func(weights[t], weights[t-1], self.rets[t-1], self.cov[t-1])
            constr += self.single_period_constraints(weights[t])  # Concatenate constraints

        constr += [weights[0] == self.start_weights]  # first weights are fixed at known current portfolio

        prob = cp.Problem(cp.Maximize(objective), constr) # Construct maximization problem
        prob.solve(verbose=verbose)
        opt_var = weights.value

        if verbose is True:
            print("Shape of var: ", opt_var.shape)
            temp_df = pd.DataFrame(opt_var).round(3)
            temp_df['sum_weights'] = np.sum(opt_var, axis=1)
            print(temp_df)

        return opt_var[1:]  # Discard first row which is not a variable.

    def _port_ret(self, weights, rets):
        """
        Compute portfolio return at a specific point in time.

        Takes a vector of weights and another vector of returns at time t.
        Returns the dot product -> Scalar.
        """
        port_ret_ = rets @ weights
        return port_ret_

    def _trading_cost(self, current_weights, prev_weights):
        """
        Using current and previous portfolio weights computes the trading costs as scalar value.

        Assumes no cost in trading risk-free asset and thus discards last value in delta_weight.
        """
        delta_weight = current_weights - prev_weights
        delta_weight = delta_weight[:-1]  # No costs associated with risk-free asset
        trading_cost = self.kappa1 * cp.abs(delta_weight) + self.kappa2 * cp.square(delta_weight)  # Vector of trading costs per asset

        return cp.sum(trading_cost)

    def _holding_cost(self, weights):
        """
        Portfolio holding costs.
        """
        holding_cost_ = self.rho1 * cp.abs(weights) + self.rho2 * cp.square(weights)
        return cp.sum(holding_cost_)

    def _gamma_from_drawdown_control(self):
        """
        Compute gamma (risk-aversion parameter) using drawdown control.

        """
        # From all previous total portfolio values get the highest one
        previous_peak = np.max(self.prev_port_vals)

        drawdown_t = 1 - self.prev_port_vals[-1] / previous_peak
        denom = np.max([self.max_drawdown - drawdown_t, self.eps])  # If drawdown limit is breached use a number very close to zero
        gamma = self.gamma_0 * self.max_drawdown / denom

        return gamma

    def _port_risk_control(self, weights, cov):
        """
        Computes portfolio risk parameter. Currently set to portfolio variance.
        """
        port_var = cp.quad_form(weights, cov)  # CVXPY method for doing: w^T @ COV @ w
        return port_var

if __name__ == "__main__":
    df_ret = load_returns()
    cov = df_ret.cov().to_numpy()

    ret_pred = df_ret.iloc[-15:].to_numpy()
    weights = np.zeros(len(cov))
    weights[-1] = 1.
    prev_port_vals = np.array([100, 105,110,100,105,120,130,150,135,160])

    model = MPC(ret_pred, cov, prev_port_vals)
    weigths = model.cvxpy_solver()


