import warnings

import numpy as np
import pandas as pd;
from matplotlib import pyplot as plt

from models.finance.backtest import Backtester
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM
from utils.data_prep import load_returns, load_logreturns, load_prices

np.seterr(divide='ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

figsize = (20 ,10)
def plot_port_weights(weights, df, start=None, constraints=None, show=True, savefig=None):
    # Prepare data
    df.dropna(inplace=True)
    df = df.iloc[-len(weights):]
    if not start == None:
        df = df.loc[start:]
        #weights = weights[-len(df):]

    df = df / df.iloc[0] * 100

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=figsize)

    ax.stackplot(df.index, weights.T, labels=df.columns)

    if constraints in ['long_only', 'LLO', 'lo']:
        ax.set_ylim(top=1.)

    ax.set_xlim(df.index[0], df.index[-1])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel('Asset weight')
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    if show == True:
        plt.show()

    return fig, ax

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

    ax[0].plot(df.index, df)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('log $P_t$')

    ax[1].stackplot(df.index, weights.T, labels=df.drop('port_val',axis=1).columns)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[1].set_ylabel('Asset weight')

    ax[1].set_xlim(df.index[0], df.index[-1])

    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    if show:
        plt.show()

    return fig, ax


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
    df = load_prices(out_of_sample=out_of_sample)  # Price - not returns
    df_ret = load_returns(out_of_sample=out_of_sample)  # Actual returns - used to test performance of trading strategy
    df_logret = load_logreturns(out_of_sample=out_of_sample)
    X = df_logret["S&P 500 "]


    # Uncomment this section to perform new backtest - generating forecast distributions
    # Leave commented to used existing preds and covariances from file
    #preds, cov = backtester.rolling_preds_cov_from_hmm(X, df_logret, model, window_len=1700, shrinkage_factor=(0.3, 0.3), verbose=True)
    #np.save(path + 'preds_' + sample_type + '.npy', preds)
    #np.save(path + 'cov_' + sample_type + '.npy', cov)

    # Leave uncomented to use forecast distributions from file
    preds = np.load(path + 'preds_'+ sample_type + '.npy')
    cov = np.load(path + 'cov_' + sample_type + '.npy')

    # Use forecast distribution to test trading strategy
    weights, port_val, gamma = backtester.backtest_mpc(df_ret, preds, cov, short_cons='LO')
    np.save(path + 'weights_' + sample_type + '.npy', weights)
    np.save(path + 'port_val_' + sample_type + 'npy', port_val)
    np.save(path + 'gamma_' + sample_type + '.npy', gamma)

    # Leave uncommented to use previously tested trading strategy from file
    #port_val = np.load(path + 'port_val.npy')
    #weights = np.load(path + 'mpc_weights.npy')

    # Compare portfolio to df with benchmarks
    metrics = backtester.performance_metrics(df, port_val, compare_assets=True)
    print(metrics)


    # Plotting
    df = df.iloc[-len(port_val):]

    save = False
    if save == True:
        metrics.round(4).to_latex(path + 'asset_performance_' + sample_type +  '.tex')
        plot_performance(df, port_val, weights, savefig='/' + model_str + 'performance.png')
    else:
        plot_performance(df, port_val, weights, savefig=None)








    # print('transaction costs:', (1-backtester.trans_cost).prod())
    # print('highest trans cost', backtester.trans_cost.max())