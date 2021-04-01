import copy

import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from utils.data_prep import load_long_series_logret
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM


def train_rolling_window(logret, mle, jump, window_lens=[1700], n_lags=100):
    n_obs = len(logret)
    df = pd.DataFrame()  # Create empty df to store data in

    # Set attributes to be saved in rolling window
    cols = {
        '$\mu_1$': [], '$\mu_2$': [],
        '$\sigma_1$': [], '$\sigma_2$': [],
        '$q_{11}$': [], '$q_{22}$': [], 'timestamp': [],
        '$\phi_1$': [], '$\phi_2$':[]}

    cols_1 = {f'lag_{i}': [] for i in range(n_lags)}

    cols.update(cols_1)


    # Compute models params for each window length
    for window_len in tqdm.tqdm(window_lens):
        # Load empty dictionaries for each run
        data = {'jump': copy.deepcopy(cols),  # copy column names from variable cols above
                'mle': copy.deepcopy(cols)
                }

        # Loop through data and fit models at each time step
        for t in tqdm.tqdm(range(window_len, n_obs)):
            # Slice data into rolling sequences
            rolling = logret.iloc[t - window_len: t]

            # Fit models to rolling data
            mle.fit(rolling, sort_state_seq=True, verbose=True)
            jump.fit(rolling, get_hmm_params=True, sort_state_seq=True, verbose=True)

            ## Lav simulering her

            #simulation = mle.sample(n_samples=2000)[0]  #>1000  ##Generates 2000 returns to step 1
            #simulation_squared = simulation**2 # Squaring return
            ### regn acf på simualtion squared (brug pakke stats.models eller lign)

            #simulation_jump = jump.sample()
            # Save data
            data['jump']['$\mu_1$'].append(jump.mu[0])
            data['jump']['$\mu_2$'].append(jump.mu[1])
            data['jump']['$\sigma_1$'].append(jump.std[0])
            data['jump']['$\sigma_2$'].append(jump.std[1])
            data['jump']['$q_{11}$'].append(jump.tpm[0, 0])
            data['jump']['$\phi_1$'].append(jump.stationary_dist[0])
            data['jump']['$\phi_2$'].append(jump.stationary_dist[1])


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
            data['mle']['$\phi_1$'].append(mle.stationary_dist[0])
            data['mle']['$\phi_2$'].append(mle.stationary_dist[1])

            # Test if more than 1 state is detected
            if len(mle.tpm) > 1:
                data['mle']['$q_{22}$'].append(mle.tpm[1, 1])
            else:
                data['mle']['$q_{22}$'].append(0)

            # Add timestamps
            data['jump']['timestamp'].append(rolling.index[-1])
            data['mle']['timestamp'].append(rolling.index[-1])

            ## Lav loop til at inkluder lag 1 til 100
            for lag in range(n_lags):
                data['mle'][f'lag_{lag}'].append(mle.squared_acf(lag=lag))
                data['jump'][f'lag_{lag}'].append(jump.squared_acf(lag=lag))

                # Lav kode der assigner ACF for hver lag der er simuleret foroven.
                ## Bare skriv navnet på ACF variablen fra simuleret også [i] eksempelvis acf[i]
        # Add model name and window len to data and output a dataframe
        for model in data.keys():
            df_temp = pd.DataFrame(data[model])
            df_temp['model'] = model
            df_temp['window_len'] = window_len
            df = df.append(df_temp)

    return df



if __name__ == '__main__':
    # Load SP500 logrets
    logret = load_long_series_logret()

    # Instantiate HMM models
    mle = EMHiddenMarkov(n_states=2, epochs=10, max_iter=100, random_state=42)
    jump = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                   epochs=2, max_iter=30, random_state=42)

    logret = logret[10000:15000]  # Reduce sample size to speed up training

    df = train_rolling_window(logret, mle, jump, window_lens=[1000], n_lags=20)


    # Group data first by window len and the by each mode. Returns mean value of each remaining parameter
    data_table = df.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])

    print(data_table)

    # Save results
    save = True
    if save == True:
        path = '../../analysis/stylized_facts/output_data/'
        df.to_csv(path + 'rolling_estimations.csv', index=False)
        data_table.round(4).to_latex(path + 'test.tex', escape=False)

