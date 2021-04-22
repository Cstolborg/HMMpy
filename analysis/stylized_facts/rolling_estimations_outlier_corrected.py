import copy

import pandas as pd;
from scipy import stats

pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import statsmodels.api as sm
from utils.data_prep import load_long_series_logret, DataPrep
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM
import warnings
warnings.filterwarnings("ignore")

from analysis.stylized_facts.rolling_estimations import train_rolling_window

if __name__ == '__main__':
    # Load log returns from SP500
    data = DataPrep()
    logrets_outlier = data.load_long_series_logret(outlier_corrected=True, threshold=4)

    # Instantiate HMM models
    mle = EMHiddenMarkov(n_states=2, epochs=10, max_iter=100, random_state=42)
    jump = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                   epochs=20, max_iter=30, random_state=42)

    #logret = logret[13000:15000]  # Reduce sample size to speed up training

    df = train_rolling_window(logrets_outlier, mle, jump, window_lens=[1700], n_lags=100,
                              outlier_corrected=False, absolute_moments=True, n_sims=5000)

    # Group data first by window len and the by each mode. Returns mean value of each remaining parameter
    data_table = df.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])
    print(data_table)

    # Save results
    save = True
    if save == True:
        path = '../../analysis/stylized_facts/output_data/'
        df.to_csv(path + 'rolling_estimations_outlier_corrected_abs.csv', index=False)
