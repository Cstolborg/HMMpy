import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models.finance.backtest import Backtester
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM
from utils.data_prep import load_returns, load_logreturns, load_prices, DataPrep

np.seterr(divide='ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

if __name__ == "__main__":
    # Set path, model to test and in-sample vs. out-of-sample
    model_str = 'mle'
    path = '../../analysis/portfolio_exercise/output_data/' + model_str + '/'
    out_of_sample = False

    sample_type = 'oos' if out_of_sample is True else 'is'  # Used to specify suffix in file names

    # Instantiate models to test and backtester
    if model_str == 'mle':
        model = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)
    elif model_str == 'jump':
        model = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                       epochs=20, max_iter=30, random_state=42)

    backtester = Backtester()

    # Get data - logreturns is used in HMM model
    data = DataPrep(out_of_sample=out_of_sample)
    X = data.logrets["S&P 500 "]
    window_len = 1500

    # Uncomment this section to perform new backtest - generating forecast distributions
    # Leave commented to used existing preds and covariances from file
    preds, cov = backtester.rolling_preds_cov_from_hmm(X, data.logrets, model, window_len=window_len, shrinkage_factor=(0.3, 0.3), verbose=True)
    np.save(path + 'preds_' + sample_type + str(window_len) + '.npy', preds)
    np.save(path + 'cov_' + sample_type + str(window_len) + '.npy', cov)