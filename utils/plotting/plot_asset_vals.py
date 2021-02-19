import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def plot_asset_vals(df, port_val, show=True):
    df.dropna(inplace=True)
    df = df.iloc[1500:]
    df['port_val'] = port_val
    df = df / df.iloc[0] * 100

    df.plot()
    if show == True:
        plt.show()




if __name__ == '__main__':
    df = pd.read_csv('../../data/price_series.csv', index_col='Time')
    port_val = np.load('../../data/port_val.npy')
    plot_asset_vals(df, port_val)
