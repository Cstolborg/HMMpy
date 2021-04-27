import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.data_prep import load_prices, DataPrep

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

def plot_asset_vals(df, start=None, show=True, savefig=False):
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
    ax.legend(lineObjects, custom_labels, loc='upper left')#,bbox_to_anchor=(1.05, 1))#, loc='upper left')

    #ax.set_yscale('log')
    ax.set_ylabel(r'$P_t$')
    ax.set_xlim(df.index[0], df.index[-1])
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    if show:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    data_oos = DataPrep(out_of_sample=True)
    data_is = DataPrep(out_of_sample=False)
    start = None

    metrics_oos = compute_asset_metrics(data_oos.prices, start=start).round(4)
    metrics_insample = compute_asset_metrics(data_is.prices, start=start).round(4)
    print(metrics_oos)
    print(metrics_insample)

    save = True
    if save:
        metrics_oos.to_latex('../../analysis/portfolio_exercise/output_data/asset_performance.tex')
        plot_asset_vals(data_oos.prices.iloc[1000:], start=start, savefig='asset_vals_oos.png')
        plot_asset_vals(data_is.prices, start=start, savefig='asset_vals_insample')
    else:
        plot_asset_vals(data_oos.prices, start=start, savefig=None)
        plot_asset_vals(data_is.prices, start=start, savefig=None)

