import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.data_prep import load_data

import warnings
warnings.filterwarnings('ignore')


figsize = (20 ,10)

def plot_asset_vals(df, port_val, show=True, save=False):
    df.dropna(inplace=True)
    df = df.iloc[-len(port_val):]
    df['port_val'] = port_val
    df = df / df.iloc[0] * 100

    df.plot(figsize=figsize)
    #df[['MSCI World', 'S&P 500 ', 'Barclays US Treasury','DAX ', 'port_val']].plot()

    plt.tight_layout()

    if save == True:
        plt.savefig('../../analysis/portfolio_exercise/images/asset_vals')
    if show == True:
        plt.show()

def plot_port_weights(weights, index, labels, show=True, save=False):
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(index, weights.T, labels=labels)

    #ax.set_xlim(start, end)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')#, fontsize='x-small')
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
    fig, ax = plt.subplots(nrows = 2, ncols=1, sharex=True, figsize=figsize)

    ax[0].plot(df.index, df)
    ax[0].set_yscale('log')

    ax[1].stackplot(df.index, weights.T, labels=df.drop('port_val',axis=1).columns)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if show:
        plt.show()
    if save:
        plt.savefig('../../analysis/portfolio_exercise/images/port_performance')

    return fig, ax


if __name__ == '__main__':
    df = load_data()
    port_val = np.load('../../analysis/portfolio_exercise/output_data/port_val.npy')
    mpc_weights = np.load('../..//analysis/portfolio_exercise/output_data/mpc_weights.npy')


    #plot_asset_vals(df, port_val, save=True)
    #plot_port_weights(mpc_weights, df.index.values, labels=df.columns, save=True)
    plot_performance(df, port_val, mpc_weights)