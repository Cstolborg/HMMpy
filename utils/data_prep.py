import numpy as np
import pandas as pd

""" TODO
Data quality is poor, several entries are null. Get better data or impute it.

"""

def load_data_get_ret(path='../data/price_series.csv'):
    df = pd.read_csv(path, index_col='Time')
    df.dropna(inplace=True)
    df_ret = df.pct_change()
    df_ret.dropna(inplace=True)

    return df_ret

def load_data_get_logret(path='../data/price_series.csv'):
    df = pd.read_csv(path, index_col='Time')
    df.dropna(inplace=True)
    df_ret = np.log(df) - np.log(df.shift(1))
    df_ret.dropna(inplace=True)

    return df_ret

def get_cov_mat(df_ret):
    return df_ret.cov()


if __name__ == '__main__':
    df = load_data_get_ret()