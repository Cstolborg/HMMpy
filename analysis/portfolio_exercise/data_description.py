import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.data_prep import load_data

import warnings
warnings.filterwarnings('ignore')


figsize = (20 ,10)


def compute_asset_metrics(df,start=None):
    """Compute performance metrics for a given portfolio/asset"""
    # Prepare data
    df.dropna(inplace=True)
    if not start == None:
        df = df.loc[start:]

    df_ret = df.pct_change().dropna()

    # Annual returns, std
    n_years = len(df) / 252

    ret = df_ret.drop('T-bills rf', axis=1)
    cagr = ((1 + ret).prod(axis=0)) ** (1 / n_years) - 1
    std = ret.std(axis=0, ddof=1) * np.sqrt(252)

    excess_ret = df_ret.subtract(df_ret['T-bills rf'], axis=0).drop('T-bills rf', axis=1)
    excess_cagr = ((1 + excess_ret).prod(axis=0)) ** (1 / n_years) - 1
    excess_std = excess_ret.std(axis=0, ddof=1) * np.sqrt(252)
    sharpe = excess_cagr / excess_std

    df = df.drop('T-bills rf', axis=1)
    peaks = df.cummax(axis=0)
    drawdown = -(df - peaks) / peaks
    max_drawdown = drawdown.max(axis=0)
    calmar = excess_cagr / max_drawdown

    metrics = {#'Return': cagr,
               #'Std': std,
               'Return': excess_cagr,
               'Std': excess_std,
               'Sharpe': sharpe,
               'Max drawdown': max_drawdown,
               'Calmar ratio': calmar}

    metrics = pd.DataFrame(metrics)

    return metrics


def plot_asset_vals(df, start=None, show=True, save=False):
    # Prepare data
    df.dropna(inplace=True)
    if not start == None:
        df = df.loc[start:]

    df = df / df.iloc[0] * 100  # Index data to 100

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    lineObjects = ax.plot(df.index, df)
    custom_labels = df.columns

    # Labels have to be assigned this way if one wants to avoid a loop
    ax.legend(lineObjects, custom_labels,bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_yscale('log')
    ax.set_ylabel('log $P_t$')
    ax.set_xlim(df.index[0], df.index[-1])
    plt.tight_layout()

    if save:
        plt.savefig('../../analysis/portfolio_exercise/images/asset_vals.png')
    if show:
        plt.show()

    return fig, ax

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





if __name__ == '__main__':
    df = load_data()
    port_val = np.load('../../analysis/portfolio_exercise/output_data/port_val.npy')
    weights = np.load('../..//analysis/portfolio_exercise/output_data/mpc_weights.npy')

    start = '2000-09-01'

    metrics = compute_asset_metrics(df, start=start).round(4)
    print(metrics)

    save = True
    if save:
        metrics.to_latex('../../analysis/portfolio_exercise/output_data/asset_performance.tex')
        plot_asset_vals(df, start=start, save=True)
        plot_port_weights(weights, df, start=None, constraints='long_only', save=True)
        #plot_performance(df, port_val, weights, save=True)
    else:
        plot_asset_vals(df, start=start, save=False)
        plot_port_weights(weights, df, start=None, constraints='long_only', save=False)
        # plot_performance(df, port_val, weights, save=False)
