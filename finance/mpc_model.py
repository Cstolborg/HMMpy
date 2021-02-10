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
    """

    def __init__(self, ret_pred, covariances, max_drawdown=0.4, gamma_0=5, kappa1=0.004,
                 rho2=0.0005, max_holding=0.4):

        self.max_drawdown = max_drawdown
        self.gamma_0 = gamma_0
        self.kappa1 = kappa1
        self.rho2 = rho2
        self.max_holding = max_holding
        self.ret_pred = ret_pred
        self.cov = covariances

        self.n_assets = self.cov.shape[0]
        self.n_preds = len(self.ret_pred)
        self.weights = np.zeros(shape=(self.n_preds, self.n_assets))

    def port_ret(self):
        port_ret_ = self.ret_pred @ self.weights

        vec = self.ret_pred[0] @ self.weights[0]
        return port_ret_, vec



if __name__ == "__main__":
    sampler = SampleHMM(n_states=2, random_state=1)
    X, true_states = sampler.sample(15)

    df_ret = load_data_get_ret()
    cov = get_cov_mat(df_ret).values

    ret_pred = df_ret.iloc[-15:].to_numpy()

    model = MPC(ret_pred, cov_np)
    model.port_ret()







