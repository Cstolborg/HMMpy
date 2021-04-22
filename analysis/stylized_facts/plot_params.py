import copy
import warnings

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from statsmodels.tsa.stattools import acf

from analysis.stylized_facts.rolling_estimations import train_rolling_window
from utils.data_prep import load_long_series_logret
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM

from models.hidden_markov.hmm_jump import JumpHMM
from utils.data_prep import DataPrep
from analysis.stylized_facts.plot_rolling_estimations import plot_rolling_parameters


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

def train_rolling_window(logret, mle, jump, window_lens=[1700], n_lags=100, n_sims=5000,
                         get_acf=True, absolute_moments=False, outlier_corrected=False):
    n_obs = len(logret)
    df = pd.DataFrame()  # Create empty df to store data in

    # Set attributes to be saved in rolling window
    cols = {
        '$\mu_1$': [], '$\mu_2$': [],
        '$\sigma_1$': [], '$\sigma_2$': [],
        '$q_{11}$': [], '$q_{22}$': [], 'timestamp': [],
        '$\pi_1$': [], '$\pi_2$':[],
        'mean': [], 'variance': [],
        'skewness': [], 'excess_kurtosis': []
        }

    if get_acf is True:
        cols_1 = {f'lag_{i}': [] for i in range(n_lags)}
        cols.update(cols_1)

    # Compute models params for each window length
    for window_len in window_lens:
        print(f'Training on window length = {window_len}')
        # Load empty dictionaries for each run
        data = {'jump': copy.deepcopy(cols),  # copy column names from variable cols above
                'mle': copy.deepcopy(cols)
                }

        # Loop through data and fit models at each time step
        for t in tqdm.tqdm(range(window_len, n_obs)):
            # Slice data into rolling sequences
            rolling = logret.iloc[t - window_len: t]

            # Remove all observations with std's above 4
            if outlier_corrected is True:
                rolling = rolling[(np.abs(stats.zscore(rolling)) < 4)]

            # Fit models to rolling data
            mle.fit(rolling, sort_state_seq=True, get_hmm_params=True, verbose=True, feature_set='feature_set_3')
            jump.fit(rolling, sort_state_seq=True, get_hmm_params=True, verbose=True, feature_set='feature_set_2')

            # Save data
            data['jump']['$\mu_1$'].append(jump.mu[0])
            data['jump']['$\mu_2$'].append(jump.mu[1])
            data['jump']['$\sigma_1$'].append(jump.std[0])
            data['jump']['$\sigma_2$'].append(jump.std[1])
            data['jump']['$q_{11}$'].append(jump.tpm[0, 0])
            data['jump']['$\pi_1$'].append(jump.stationary_dist[0])
            data['jump']['$\pi_2$'].append(jump.stationary_dist[1])


            # Test if more than 1 state is detected
            if len(jump.tpm) > 1:
                data['jump']['$q_{22}$'].append(jump.tpm[1, 1])
            else:
                data['jump']['$q_{22}$'].append(0)

            data['mle']['$\mu_1$'].append(mle.mu[0])
            data['mle']['$\mu_2$'].append(mle.mu[1])
            data['mle']['$\sigma_1$'].append(mle.std[0])
            data['mle']['$\sigma_2$'].append(mle.std[1])
            data['mle']['$q_{11}$'].append(mle.tpm[0, 0])
            data['mle']['$\pi_1$'].append(mle.stationary_dist[0])
            data['mle']['$\pi_2$'].append(mle.stationary_dist[1])

            # Test if more than 1 state is detected
            if len(mle.tpm) > 1:
                data['mle']['$q_{22}$'].append(mle.tpm[1, 1])
            else:
                data['mle']['$q_{22}$'].append(0)

            # Add timestamps
            data['jump']['timestamp'].append(rolling.index[-1])
            data['mle']['timestamp'].append(rolling.index[-1])

            ## Simulate data to test temporal and distributionalt properties
            mle_simulation = mle.sample(n_samples=n_sims)[0]  # Simulate returns
            jump_simulation = jump.sample(n_samples=n_sims)[0]

            if absolute_moments is True:
                mle_simulation = np.abs(mle_simulation)
                jump_simulation = np.abs(jump_simulation)

            if get_acf is True:
                mle_simulation_abs = np.abs(mle_simulation)
                jump_simulation_abs = np.abs(jump_simulation)
                jump_acf_abs_simulated = acf(jump_simulation_abs, nlags=n_lags)[1:]
                mle_acf_abs_simulated = acf(mle_simulation_abs, nlags=n_lags)[1:]

                for lag in range(n_lags):
                    data['mle'][f'lag_{lag}'].append(mle_acf_abs_simulated[lag])
                    data['jump'][f'lag_{lag}'].append(jump_acf_abs_simulated[lag])

            data['mle']['mean'].append(np.mean(mle_simulation))
            data['mle']['variance'].append(np.var(mle_simulation, ddof=1))
            data['mle']['skewness'].append(stats.skew(mle_simulation))
            data['mle']['excess_kurtosis'].append(stats.kurtosis(mle_simulation)) # Excess kurtosis

            data['jump']['mean'].append(np.mean(jump_simulation))
            data['jump']['variance'].append(np.var(jump_simulation, ddof=1))
            data['jump']['skewness'].append(stats.skew(jump_simulation))
            data['jump']['excess_kurtosis'].append(stats.kurtosis(jump_simulation)) # Excess kurtosis

        # Add model name and window len to data and output a dataframe
        for model in data.keys():
            df_temp = pd.DataFrame(data[model])
            df_temp['model'] = model
            df_temp['window_len'] = window_len
            df = df.append(df_temp)

    df.set_index('timestamp', inplace=True)

    return df


if __name__ == '__main__':
    data = DataPrep()
    logrets = data.load_long_series_logret()
    logrets = logrets.loc['1997-03-20':'2005-03-31']

    mle = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                   epochs=20, max_iter=30, random_state=42)
    jump = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                   epochs=20, max_iter=30, random_state=42)


    df = train_rolling_window(logrets, mle, jump, window_lens=[1700], n_lags=100, get_acf=True,
                              absolute_moments=True, outlier_corrected=False, n_sims=5000)



    plot_rolling_parameters(df, model='jump', savefig=None)
    plot_rolling_parameters(df, model='mle', savefig=None)

