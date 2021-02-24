import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


def plot_asset_vals(df, port_val, show=True):
    df.dropna(inplace=True)
    df = df.iloc[-len(port_val):]
    df['port_val'] = port_val
    df = df / df.iloc[0] * 100

    #df.plot()
    df[['MSCI World', 'S&P 500 ', 'Barclays US Treasury','DAX ', 'port_val']].plot()

    plt.tight_layout()
    if show == True:
        plt.show()

def plot_port_weights(weights, index, start, end, labels, show=True):
    fig, ax = plt.subplots(figsize=(12,7))
    ax.stackplot(range(len(weights)), weights.T, labels=labels)
    #ax.set_xticks(index)

    #ax.set_xlim(start, end)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')#, fontsize='x-small')
    plt.tight_layout()

    if show == True:
        plt.show()
    return fig, ax


if __name__ == '__main__':
    df = pd.read_csv('../../data/price_series.csv', index_col='Time')
    port_val = np.load('../../data/port_val.npy')
    mpc_weights = np.load('../../data/mpc_weights.npy')

    df.dropna(inplace=True)
    df = df.iloc[1500:]
    start = datetime.strptime(df.index[0], '%Y-%m-%d')
    end = datetime.strptime(df.index[-1], '%Y-%m-%d')

    plot_asset_vals(df, port_val)
    plot_port_weights(mpc_weights, df.index.values, start, end, labels=df.columns)
