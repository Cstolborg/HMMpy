import numpy as np; np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

from hmm_models.hmm_gaussian_em import EMHiddenMarkov
from utils.data_transformation import load_data_get_ret, load_data_get_logret

import warnings
warnings.filterwarnings('ignore')

""" TODO next
Have method of generating mu and cov. Need to verify computations.
Next step is to use those with posteriors to forecast returns.

"""

class Backtester:

    def __init__(self, model, X, df_rets):
        self.model = model
        self.X = X
        self.df_rets = df_rets

    def get_asset_dist(self, df, state_sequence):
        df['state_sequence'] = state_sequence
        groupby_state = df.groupby('state_sequence')

        log_mu, log_cov = groupby_state.mean(), groupby_state.cov()

        mu = np.zeros(shape=(self.model.n_states, self.df_rets.shape[1]))
        covariance = np.zeros(shape=(self.df_rets.shape[1], self.df_rets.shape[1]))
        cov = []

        for s in range(self.model.n_states):
            mu[s], cov_temp = self.logcov_to_cov(log_mu.iloc[s], log_cov.loc[s])

            cov.append(cov_temp)

        return mu, cov

    def logcov_to_cov(self, log_mu, log_cov):
        diag = np.diag(log_cov)
        mu = np.exp(log_mu + np.diag(log_cov)/2) - 1
        x1 = np.outer(mu, mu)
        x2 = np.outer(diag, diag) / 2
        cov = np.exp(x1+x2) * (np.exp(log_cov) - 1)

        return mu, cov


    def rolling_backtest_hmm(self, X, df, n_preds=15, window_length=1500):

        for t in range(window_length,len(df)):
            df_rolling = df.iloc[t-window_length:t]
            X_rolling = X.iloc[t-window_length:t]

            model.fit(X_rolling)
            state_sequence = model.decode(X_rolling)  # 1darray with most likeley state sequence
            posteriors = model.predict_proba(n_preds)  # Posterior probability of being in state j at time t
            mu, cov = self.get_asset_dist(df_rolling, state_sequence)


            self.state_sequence = state_sequence
            self.posteriors = posteriors
            self.mu, self.cov = mu, cov

            break


if __name__ == "__main__":
    df_ret = load_data_get_logret()
    df_ret["rf"] = 0
    X = df_ret["MSCI World"]

    model = EMHiddenMarkov(n_states=2, init="random", random_state=1, epochs=20, max_iter=50)

    backtester = Backtester(model, X, df_ret)

    backtester.rolling_backtest_hmm(X, df_ret)
