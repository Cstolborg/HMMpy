import numpy as np; np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import tqdm

from hmm_models.hmm_gaussian_em import EMHiddenMarkov
from utils.data_prep import load_data_get_logret

import warnings
warnings.filterwarnings('ignore')

""" TODO
In Backtester.get_asset_dist when only 1 state is present the other takes on mean
values of 0 and covariances of 0. Is this a good result ?

Sometimes a state is not vistited or only has 1 observation in which case covariance matrix will return null results.
We need a solution for this.

"""

class Backtester:
    """
    Backtester for Hidden Markov Models.

    Computes posteriors, state sequences as well as expected and forecasted returns and standard deviations.

    Parameters
    ----------
    model : hidden markov model
        Hidden Markov Model object.
    X : ndarray of shape (n_samples,)
        Times series data used to train the HMM.
    df_rets : DataFrame of shape (n_samples, n_assets)
        Times series data used when estimating expected returns and covariances.

    Attributes
    ----------
    preds : ndarray of shape (n_samples-window_len, n_preds, n_assets)
        mean predictions for each asset h time steps into the future at each time t.
    cov : ndarray of shape(n_samples-window_len, n_preds, n_assets, n_assets)
        predicted covariance matrix h time steps into the future at each time t.
    """
    def __init__(self, model, X, df_rets):
        self.model = model
        self.X = X
        self.df_rets = df_rets

        self.preds = None
        self.cov = None

        self.n_states = model.n_states
        self.n_assets = self.df_rets.shape[1]

    def get_asset_dist(self, df, state_sequence):
        """
        Compute multivariate normal distribution of all assets in each state.

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
        df['state_sequence'] = state_sequence
        groupby_state = df.groupby('state_sequence')
        log_mu, log_cov = groupby_state.mean(), groupby_state.cov()
        state_count = groupby_state.count().max(axis=1)  # Num obs in each state

        mu = np.zeros(shape=(self.n_states, self.n_assets))
        cov = np.zeros(shape=(self.n_states, self.n_assets, self.n_assets))

        # Loop through n_states present in current sample
        for s in log_mu.index:
            try:
                if state_count > 1:  # If state_count not >1, covariance will return NaN
                    mu[s], cov[s] = self.logcov_to_cov(log_mu.loc[s], log_cov.loc[s])
            except:
                print('s: ', s)
                print('log mu:')
                print(log_mu)
                print('log COV:')
                print(log_cov)

        return mu, cov

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
        mu = np.exp(log_mu + np.diag(log_cov)/2) - 1
        x1 = np.outer(mu, mu)
        x2 = np.outer(diag, diag) / 2
        cov = np.exp(x1+x2) * (np.exp(log_cov) - 1)

        return mu, cov

    def rolling_backtest_hmm(self, X, df, n_preds=15, window_len=1500, progress_bar=True, debug=False):
        """
        Backtest based on rolling windows.

        Fits a Hidden Markov model within each rolling window and computes the unconditional
        multivariate normal mixture distributions for each asset in the defined universe.

        Parameters
        ----------
        X
        df
        n_preds
        window_len
        progress_bar
        debug

        Returns
        -------
        preds : ndarray of shape (n_samples-window_len, n_preds, n_assets)
            Unconditional mean values for each asset
        cov : ndarray of shape (n_samples-window_len, n_preds, n_assets, n_assets)
            Unconditional covariance matrix at each time step t, h steps into future
        """

        # Create 3- and 4-D array to store predictions and covariances
        self.preds = np.empty(shape=(len(df) - window_len, n_preds, self.n_assets))  # 3-D array
        self.cov = np.empty(shape=(len(df) - window_len, n_preds, self.n_assets, self.n_assets))  # 4-D array

        for t in tqdm.trange(window_len, len(df)):
            # Slice data into rolling sequences
            df_rolling = df.iloc[t - window_len:t]
            X_rolling = X.iloc[t - window_len:t]

            self.model.fit(X_rolling)
            if self.model.is_fitted == False:  # Check if model is fitted
                print(f'refitting at t: {t}...')
                max_iter = self.model.max_iter
                self.model.max_iter = max_iter * 2  # Double amount of iterations
                self.model.fit(X_rolling)  # Try fitting again
                self.model.max_iter = max_iter  # Reset max_iter back to user-input
                if self.model.is_fitted == False:
                    print(f'NOT FITTED at t: {t} -- mu {self.model.mu} -- tpm {np.diag(self.model.tpm)}')

            state_sequence = self.model.decode(X_rolling)  # 1darray with most likely state sequence

            # Posterior probability of being in state j at time t
            posteriors = self.model.predict_proba(n_preds)  # 2-D array of shape (n_preds, n_states)
            mu, cov = self.get_asset_dist(df_rolling, state_sequence)  # shapes (n_states, n_assets), (n_states, n_assets, n_assets)
            preds = np.inner(mu.T, posteriors).T  # shape (n_preds, n_assets)

            cov_x1 = np.inner(posteriors, cov.T)  # shape (n_preds, n_assets, n_assets)
            cov_x2 = preds - mu[:, np.newaxis]  # shape (n_states, n_preds)
            cov_x3 = np.einsum('ijk,ijk->ij', cov_x2, cov_x2)  # Equal to np.sum(X**2, axis=-1)
            cov_x4 = np.einsum('ij,ij->i', cov_x3.T, posteriors)  # Equal to np.sum(X3*posteriors, axis=1)
            cov_x5 = cov_x1 + cov_x4[:, np.newaxis, np.newaxis]  # shape (n_samples-window_len, n_preds, n_assets, n_assets)

            if np.any(np.isnan(preds)) == True:
                print('t: ', t)
                print('posteriors: ', posteriors)
                print(preds)
                print(f'NaNs in cov: {np.any(np.isnan(cov_x5))} -- NaNs in posteriors: {np.any(np.isnan(posteriors))} ')

                # TODO this is a quick fix and must be solved better in cases where no solutins are found!
                self.preds[t - window_len] = self.preds[t - window_len - 1]
                self.cov[t - window_len] = self.cov[t - window_len - 1]
            else:
                self.preds[t - window_len] = preds
                self.cov[t - window_len] = cov_x5

            if debug == True:
                self.state_sequence = state_sequence
                self.posteriors = posteriors
                self.mu, self.covariances = mu, cov
                self.cov_x1, self.cov_x2, self.cov_x3, self.cov_x4 = cov_x1, cov_x2, cov_x3, cov_x4

        return self.preds, self.cov


if __name__ == "__main__":
    df_ret = load_data_get_logret()
    #df_ret = df_ret.iloc[-1530:]
    X = df_ret["MSCI World"]

    model1 = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=50)
    backtester = Backtester(model1, X, df_ret)

    backtester.rolling_backtest_hmm(X, df_ret, debug=True)

    np.save('../data/rolling_preds.npy', backtester.preds)
    np.save('../data/rolling_cov.npy', backtester.cov)