import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hmmpy.finance.backtest import Backtester
from hmmpy.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from hmmpy.hidden_markov.hmm_jump import JumpHMM
from hmmpy.utils.data_prep import DataPrep

np.seterr(divide='ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

figsize = (15 ,7)
def plot_port_weights(weights, constraints=None, savefig=None):
    # Divide weights into positive and negative values
    weights_neg, weights_pos = weights.clip(upper=0.), weights.clip(lower=0.)

    # Plotting
    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=figsize)

    ax.stackplot(weights_pos.index, weights_pos.T, labels=weights.columns)
    ax.stackplot(weights_neg.index, weights_neg.T)

    if constraints in ['long_only', 'LO']:
        ax.set_ylim(top=1.)

    ax.set_xlim(weights.index[0], weights.index[-1])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=weights.shape[1]//2, fontsize=12)
    ax.set_ylabel('Asset weight')
    ax.tick_params('x', labelrotation=15)
    plt.tight_layout()

    fig.subplots_adjust(bottom=0.25)
    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

def plot_performance(df, port_val, weights, start=None, show=True, savefig=None):
    # Prepare data
    df.dropna(inplace=True)
    df = df.iloc[-len(port_val):]
    df['port_val'] = port_val

    if not start == None:
        df = df.loc[start:]
        weights = weights[-len(df):]

    df = df / df.iloc[0] * 100

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows = 2, ncols=1, sharex=True, figsize=figsize)

    ax[0].plot(df.index, df['port_val'])
    #ax[0].set_yscale('log')
    ax[0].set_ylabel('log $P_t$')

    ax[1].stackplot(df.index, weights.T, labels=df.drop('port_val',axis=1).columns)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[1].set_ylabel('Asset weight')

    ax[1].set_xlim(df.index[0], df.index[-1])

    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

if __name__ == "__main__":
    # Set path, model to test and in-sample vs. out-of-sample
    model_str = 'mle'
    path = '../../analysis/portfolio_exercise/output_data/' + model_str + '/'
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

    holding_costs = 0.001
    holding_costs_rf = 0.0000
    max_holding = 0.4
    trans_costs = 0.0040

    mpc.backtest_mpc(data.rets, preds, cov, n_preds=15, short_cons='LO',
                               kappa1=trans_costs, max_holding=max_holding, max_holding_rf=1.,
                               rho2=holding_costs, rho_rf=holding_costs_rf, gamma_0=5,
                               max_drawdown=0.1)

    weights = pd.DataFrame(mpc.weights, columns=data.prices.columns, index=data.prices.index[-len(mpc.weights):])

    equal_weigthed.backtest_equal_weighted(data.rets.iloc[1000:], rebal_freq='M')


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
        plt.plot(data.prices.index[-len(mpc.weights):], equal_weigthed.port_val, label='1/n')
        plt.legend()
        plt.show()
