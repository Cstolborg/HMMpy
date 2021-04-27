import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models.finance.backtest import Backtester
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM
from utils.data_prep import DataPrep

np.seterr(divide='ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)


if __name__ == "__main__":
    # Set path, model to test and in-sample vs. out-of-sample
    model_str = 'mle'
    path = './output_data/' + model_str + '/'
    out_of_sample = True
    sample_type = 'oos' if out_of_sample is True else 'is'  # Used to specify suffix in file names

    # Instantiate models to test and backtester
    if model_str == 'mle':
        model = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)
    elif model_str == 'jump':
        model = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                       epochs=20, max_iter=30, random_state=42)

    mpc = Backtester()
    equal_weigthed = Backtester()

    # Get data - logreturns is used in HMM model
    data = DataPrep(out_of_sample=out_of_sample)

    # Load model predictions from file - computed in mpc_preds_cov.py
    preds = np.load(path + 'preds_'+ sample_type + '.npy')
    cov = np.load(path + 'cov_' + sample_type + '.npy')

    holding_costs = 0.0000
    max_holding = 0.2
    trans_costs = 0.0010
    gammas = [1, 3, 5, 10, 15, 25]
    #constraints = [('LO', 0.1), ('LO', 0.15), ('LO', 1000)]
    #constraints = [('LLO', 0.1), ('LLO', 0.15), ('LLO', 1000), ('LS', 1000)]
    constraints = [('LS', 0.1), ('LS', 0.15), ('LS', 1000), ('LLO', 1000)]

    df = mpc.mpc_gammas_shortcons(gammas, constraints,
                            data, preds, cov, n_preds=15, max_holding_rf=1.,
                            max_leverage=2.0, trans_costs=trans_costs, holding_costs=holding_costs,
                            max_holding=max_holding)

    df.to_csv(path + 'frontiers_ls.csv', index=False)
    #df = pd.read_csv(path + 'frontiers_llo.csv')
    metrics = mpc.mulitple_port_metrics(df_port_val=df)

    print(metrics)

    #equal_weigthed.backtest_equal_weighted(data.rets, rebal_freq='M')