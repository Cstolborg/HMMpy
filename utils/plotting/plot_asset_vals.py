import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.data_prep import load_data

import warnings
warnings.filterwarnings('ignore')


def plot_asset_vals(df, port_val, show=True, save=False):
    df.dropna(inplace=True)
    df = df.iloc[-len(port_val):]
    df['port_val'] = port_val
    df = df / df.iloc[0] * 100

    df.plot()
    #df[['MSCI World', 'S&P 500 ', 'Barclays US Treasury','DAX ', 'port_val']].plot()

    plt.tight_layout()

    if save == True:
        plt.savefig('../../analysis/asset_vals')
    if show == True:
        plt.show()

def plot_port_weights(weights, index, start, end, labels, show=True, save=False):
    fig, ax = plt.subplots(figsize=(12,7))
    ax.stackplot(range(len(weights)), weights.T, labels=labels)
    #ax.set_xticks(index)

    #ax.set_xlim(start, end)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')#, fontsize='x-small')
    plt.tight_layout()

    if save == True:
        plt.savefig('../../analysis/port_weights')
    if show == True:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    df = load_data()
    port_val = np.load('../../data/port_val.npy')
    mpc_weights = np.load('../../data/mpc_weights.npy')

    df = df.iloc[-len(port_val):]
    start = df.index[0]
    end = df.index[-1]

    plot_asset_vals(df, port_val, save=True)
    plot_port_weights(mpc_weights, df.index.values, start, end, labels=df.columns, save=True)
