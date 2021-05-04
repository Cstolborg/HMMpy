import warnings

import numpy as np
import pandas as pd

from hmmpy.finance.backtest import Backtester
from hmmpy.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from hmmpy.hidden_markov.hmm_jump import JumpHMM
from hmmpy.utils.data_prep import DataPrep

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

    # Load model predictions from file - computed in mpc_preds_cov.py
    preds = np.load(path + 'preds_'+ sample_type + '.npy')
    cov = np.load(path + 'cov_' + sample_type + '.npy')

    grid = {'max_holding': [0.2, 0.35, 0.5],
            'trans_costs': [0.0001, 0.0005, 0.001, 0.004],
            'holding_costs': [0, 0.0005, 0.001],
            'holding_costs_rf': [0,0.0005, 0.001]
            }

    gridsearch_results = \
                backtester.gridsearch_mpc(grid, data.rets, preds, cov, short_cons='long_only')
    gridsearch_results.to_csv(path + 'gridsearch.csv', index=False)
    print(gridsearch_results)


    df = pd.read_csv('./output_data/mle/gridsearch_1000.csv')
    df['sharpe'] = df['return'] / df['std']

    #print(df_mle.sort_values(by=['sharpe', 'trans_costs', 'holding_costs',  'max_holding']).tail(20))
    #print(df_mle.sort_values(by=['sharpe', 'trans_costs', 'holding_costs',  'max_holding']).head(20))

