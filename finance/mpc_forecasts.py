import numpy as np; np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

from hmm_models.hmm_gaussian_em import EMHiddenMarkov
from utils.data_transformation import load_data_get_ret, load_data_get_logret

import warnings
warnings.filterwarnings('ignore')

""" TODO

mu and sigma implemented as time varying forecast in 3-D and 4-D arrays respectively.
Very advanced dimension operations going on here, needs thorough checking.
They should be ready as input to the MPC problem. Needs testing.


"""

class Backtester:

    def __init__(self, model, X, df_rets):
        self.model = model
        self.X = X
        self.df_rets = df_rets

        self.preds = None
        self.cov = None

        self.n_assets = self.df_rets.shape[1]

    def get_asset_dist(self, df, state_sequence):
        df['state_sequence'] = state_sequence
        groupby_state = df.groupby('state_sequence')
        log_mu, log_cov = groupby_state.mean(), groupby_state.cov()

        mu = np.zeros(shape=(self.model.n_states, self.n_assets))
        cov = np.zeros(shape=(self.model.n_states ,self.n_assets, self.n_assets))

        for s in range(self.model.n_states):
            mu[s], cov_temp = self.logcov_to_cov(log_mu.iloc[s], log_cov.loc[s])
            cov[s] = cov_temp

        return mu, cov

    def logcov_to_cov(self, log_mu, log_cov):
        diag = np.diag(log_cov)
        mu = np.exp(log_mu + np.diag(log_cov)/2) - 1
        x1 = np.outer(mu, mu)
        x2 = np.outer(diag, diag) / 2
        cov = np.exp(x1+x2) * (np.exp(log_cov) - 1)

        return mu, cov

    def rolling_backtest_hmm(self, X, df, n_preds=15, window_len=1500, debug=False):
        # Create 3D-array to store N predictions at each time step t
        self.preds = np.empty(shape=(len(df) - window_len, n_preds, self.n_assets))  # 3-D array
        self.cov = np.empty(shape=(len(df) - window_len, n_preds, self.n_assets, self.n_assets))  # 4-D array

        for t in range(window_len, len(df)):
            print(f"{t} out of {len(df)} iterations done")
            # Slice data into rolling sequences
            df_rolling = df.iloc[t - window_len:t]
            X_rolling = X.iloc[t - window_len:t]

            model.fit(X_rolling)
            state_sequence = model.decode(X_rolling)  # 1darray with most likely state sequence

            # Posterior probability of being in state j at time t
            posteriors = model.predict_proba(n_preds)  # 2-D array of shape (n_preds, n_states)
            mu, cov = self.get_asset_dist(df_rolling, state_sequence)  # shapes (n_states, n_assets), (n_states, n_assets, n_assets)
            preds = np.inner(mu.T, posteriors).T  # shape (n_preds, n_assets)
            self.preds[t - window_len] = preds

            cov_x1 = np.inner(posteriors, cov.T)  # shape (n_preds, n_assets, n_assets)
            cov_x2 = preds - mu[:, np.newaxis]  # shape (n_states, n_preds)
            cov_x3 = np.einsum('ijk,ijk->ij', cov_x2, cov_x2)  # Equal to np.sum(X**2, axis=-1)
            cov_x4 = np.einsum('ij,ij->i', cov_x3.T, posteriors)  # Equal to np.sum(X3*posteriors, axis=1)

            self.cov[t - window_len] = cov_x1 + cov_x4[:, np.newaxis, np.newaxis]  # shape (n_samples-window_len, n_preds, n_assets, n_assets)

            if debug == True:
                self.state_sequence = state_sequence
                self.posteriors = posteriors
                self.mu, self.covariances = mu, cov
                self.cov_x1, self.cov_x2, self.cov_x3, self.cov_x4 = cov_x1, cov_x2, cov_x3, cov_x4




if __name__ == "__main__":
    df_ret = load_data_get_logret()
    df_ret["rf"] = 0
    df_ret = df_ret.iloc[-1530:]
    X = df_ret["MSCI World"]

    model = EMHiddenMarkov(n_states=2, init="random", random_state=1, epochs=20, max_iter=50)

    backtester = Backtester(model, X, df_ret)

    backtester.rolling_backtest_hmm(X, df_ret, debug=True)
