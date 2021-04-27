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

def plot_frontier(df_metrics, ew_metrics, savefig=None):
    # Plotting
    plt.rcParams.update({'font.size': 25})
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12), sharex=False)

    #Plot ew portfolio
    axes[0].scatter(ew_metrics['excess_std'], ew_metrics['excess_return'], label='1/n')
    axes[1].scatter(ew_metrics['max_drawdown'], ew_metrics['excess_return'], label='1/n')

    for type, df_groupby in df_metrics.groupby(['short_cons', 'D_max']):
        label = f'${type[0]}_{{D_{{max}}={type[1]}}}$' if type[1] < 1. else f'${type[0]}$'
        axes[0].plot(df_groupby['excess_std'], df_groupby['excess_return'], label=label)
        axes[1].plot(df_groupby['max_drawdown'], df_groupby['excess_return'], label=label)

    for ax in axes:
        ax.set_ylabel('Annualized excess return')

    axes[0].set_xlabel('Annualized excess risk')
    axes[1].set_xlabel('Maximum drawdown')
    plt.legend(fontsize=15)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)

    plt.show()

def plot_sharpe_frontier(df_metrics, ew_metrics, savefig=None):
    gammas = [1,3,5,10,15,25]
    # Plotting
    plt.rcParams.update({'font.size': 25})
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12), sharex=True)

    #Plot ew portfolio
    axes[0].axhline(ew_metrics['sharpe'], label='1/n', color='black', ls='--')
    axes[1].axhline(ew_metrics['calmar_ratio'], label='1/n', color='black', ls='--')

    for type, df_groupby in df_metrics.groupby(['short_cons', 'D_max']):
        label = f'${type[0]}_{{D_{{max}}={type[1]}}}$' if type[1] < 1. else f'${type[0]}$'
        axes[0].plot(gammas, df_groupby['sharpe'], label=label)
        axes[1].plot(gammas, df_groupby['calmar_ratio'], label=label)

    for ax in axes:
        ax.set_xlabel('$\gamma_0$')

    axes[0].set_ylabel('Sharpe ratio')
    axes[1].set_ylabel('Calmar ratio')
    plt.legend(fontsize=15)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)

    plt.show()


def plot_sharpe_calmar(df_metrics, ew_metrics, savefig=None):
    # Plotting
    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))

    # Plot ew portfolio
    ax.scatter(ew_metrics['calmar_ratio'] ,ew_metrics['sharpe'], label='1/n', color='black', ls='--')

    for type, df_groupby in df_metrics.groupby(['short_cons', 'D_max']):
        label = f'${type[0]}_{{D_{{max}}={type[1]}}}$' if type[1] < 1. else f'${type[0]}$'
        ax.plot(df_groupby['calmar_ratio'], df_groupby['sharpe'], label=label)

    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('Calmar ratio')
    plt.legend(fontsize=15)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

def plot_port_val(df_frontiers, ew_port_val, savefig=None):

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(15, 10))

    for type, data in df_frontiers.groupby(['short_cons', 'D_max']):
        label = f'${type[0]}_{{D_{{max}}={type[1]}}}$' if type[1] < 1. else f'${type[0]}$'
        ax.plot(data['timestamp'], data['gamma_5'], label=label)

    ax.plot(data['timestamp'], ew_port_val, label='1/n')

    #ax.set_yscale('log')
    ax.set_ylabel('$P_t$')
    ax.tick_params('x', labelrotation=45)
    ax.legend(fontsize=15)

    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

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

    # Get data - logreturns is used in HMM model
    data = DataPrep(out_of_sample=out_of_sample)

    mpc = Backtester()

    df_frontiers = pd.read_csv(path + 'frontiers_llo.csv')
    df_frontiers['timestamp'] = pd.to_datetime(df_frontiers['timestamp'])
    metrics = mpc.mulitple_port_metrics(df_port_val=df_frontiers)
    print(metrics)

    # Compute equal-weighted metrics
    equal_weigthed = Backtester()
    equal_weigthed.backtest_equal_weighted(data.rets, rebal_freq='M')
    n_obs = len(df_frontiers[(df_frontiers['short_cons']== 'LLO') & (df_frontiers['D_max'] == 0.1)])
    ew_port_val = equal_weigthed.port_val[-n_obs:]
    ew_port_val = ew_port_val / ew_port_val[0] * 1000
    ew_metrics = equal_weigthed.single_port_metric(data.prices,
                                                   ew_port_val)

    save = True
    if save == True:
        path = f'{model_str}/'
        suffix = '_llo.png'
        plot_frontier(metrics, ew_metrics, savefig=path+'frontier'+suffix)
        plot_sharpe_frontier(metrics, ew_metrics, savefig=path+'sharpe_frontier'+suffix)
        plot_sharpe_calmar(metrics, ew_metrics, savefig=path+'sharpe_calmar'+suffix)
        plot_port_val(df_frontiers, ew_port_val, savefig=path+'port_vals'+suffix)
    else:
        plot_frontier(metrics, ew_metrics, savefig=None)
        plot_sharpe_frontier(metrics, ew_metrics, savefig=None)
        plot_sharpe_calmar(metrics, ew_metrics, savefig=None)
        plot_port_val(df_frontiers, ew_port_val, savefig=None)

    #equal_weigthed.backtest_equal_weighted(data.rets, rebal_freq='M')