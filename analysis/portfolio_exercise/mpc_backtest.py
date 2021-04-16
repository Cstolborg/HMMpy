import numpy as np;
from matplotlib import pyplot as plt

from models.hidden_markov.hmm_jump import JumpHMM

np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.finance.backtest import Backtester
from utils.data_prep import load_data_get_ret , load_data_get_logret, load_prices

import warnings
warnings.filterwarnings('ignore')

figsize = (20 ,10)
def plot_port_weights(weights, df, start=None, constraints=None, show=True, save=False):
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

    if save == True:
        plt.savefig('../../analysis/portfolio_exercise/images/port_weights')
    if show == True:
        plt.show()

    return fig, ax

def plot_performance(df, port_val, weights, start=None, show=True, save=False):
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

    if save:
        plt.savefig('../../analysis/portfolio_exercise/images/port_performance')
    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    path = '../../analysis/portfolio_exercise/output_data/'

    # Get log-returns - used in times series model
    df_logret = load_data_get_logret()
    X = df_logret["S&P 500 "]

    # Instantiate models to test and backtester
    mle = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)
    jump = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                   epochs=20, max_iter=30, random_state=42)
    backtester = Backtester()

    # Uncomment this section to perform new backtest - generating forecast distributions
    # Leave commented to used existing preds and covariances from file
    #preds, cov = backtester.rolling_preds_cov_from_hmm(X, df_logret, mle, window_len=1700, shrinkage_factor=(0.3, 0.3), verbose=True)
    #np.save(path + 'rolling_preds.npy', preds)
    #np.save(path + 'rolling_cov.npy', cov)

    # Leave uncomennted to use forecast distributions from file
    preds = np.load(path + 'rolling_preds.npy')
    cov = np.load(path + 'rolling_cov.npy')

    # Get actual returns - used to test performance of trading strategy
    df_ret = load_data_get_ret()

    # Use forecast distribution to test trading strategy
    #weights, port_val, gamma = backtester.backtest_mpc(df_ret, preds, cov, short_cons='LLO')
    #np.save(path + 'mpc_weights.npy', weights)
    #np.save(path + 'port_val.npy', port_val)
    #np.save(path + 'gamma.npy', gamma)

    # Leave uncommented to use previously tested trading strategy from file
    port_val = np.load(path + 'port_val.npy')
    weights = np.load(path + 'mpc_weights.npy')
    df = load_prices()  # Price - not returns

    # Compare portfolio to df with benchmarks
    metrics = backtester.performance_metrics(df, port_val, compare_assets=True)
    print(metrics)


    # Plotting
    df = df.iloc[-len(port_val):]

    save = False
    if save == True:
        metrics.round(4).to_latex(path + 'asset_performance.tex')
        plot_performance(df, port_val, weights, save=True)
    else:
        plot_performance(df, port_val, weights, save=False)








    # print('transaction costs:', (1-backtester.trans_cost).prod())
    # print('highest trans cost', backtester.trans_cost.max())