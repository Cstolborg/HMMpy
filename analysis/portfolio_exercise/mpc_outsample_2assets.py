import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hmmpy.finance.backtest import Backtester
from hmmpy.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from hmmpy.hidden_markov.hmm_jump import JumpHMM
from hmmpy.utils.data_prep import DataPrep
from analysis.portfolio_exercise.mpc_outsample import plot_port_weights, plot_performance

np.seterr(divide='ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

if __name__ == "__main__":
    # Set path, model to test and in-sample vs. out-of-sample
    model_str = 'mle'
    path = '../../analysis/portfolio_exercise/output_data/' + model_str + '/2_assets/'
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

    # Slice out S&P500 and US treasuries
    preds = preds[:, :, [3,6]]
    cov = cov[:, :, [3,6], [[3,3],[6,6]]]
    data.rets = data.rets.iloc[:, [3, 6]]

    trans_costs = 0.020
    holding_costs = 0.0001
    max_holding = 1.

    mpc.backtest_mpc(data.rets, preds, cov, n_preds=15, short_cons='LS', rf_included=False,
                               kappa1=trans_costs, max_holding=max_holding, max_holding_rf=1.,
                               rho2=holding_costs, gamma_0=5,
                               max_drawdown=0.1)

    weights = pd.DataFrame(mpc.weights, columns=data.prices.columns[[3,6]], index=data.prices.index[-len(mpc.weights):])

    equal_weigthed.backtest_equal_weighted(data.rets.iloc[-len(weights):], rebal_freq='M')


    metrics = mpc.single_port_metric(data.prices, mpc.port_val, compare_assets=True)
    print(metrics)

    #mpc.plot_port_val(data.prices, mpc.port_val, equal_weigthed.port_val, start=None, savefig=None)
    #plot_performance(data.prices, mpc.port_val, mpc.weights)

    save = False
    if save is True:
        path = f'{model_str}/'
        suffix = '_lo.png'
        plot_port_weights(weights, constraints='LO',
                          savefig=path+'weights'+suffix)
    else:
        plot_port_weights(weights, constraints='LO')

        fig, ax = plt.subplots(figsize=(12, 7))
        plt.plot(data.prices.index[-len(mpc.weights):], mpc.port_val, label='mpc')
        plt.plot(data.prices.index[-len(mpc.weights):], equal_weigthed.port_val[-len(weights):], label='1/n')
        plt.legend()
        plt.show()


