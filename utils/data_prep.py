import datetime as dt
import numpy as np
import pandas as pd

""" TODO
Daily data begins:
    MSCI Emerging: 1987-12-31
    Barclays : 1994-02-28
    T-bills rf : 1997-05-19
    DAX : 1999-01-04
    Real Estate: 1999-01-04
    Hedge funds : 2003-03-31
    PE : 2003-03-31

"""

def load_data(path='../../data/price_series.csv'):
    df = pd.read_csv(path, index_col='Time', parse_dates=True)
    df = df.drop(['Hedge Funds Global', 'Private Equity', 'DAX ',
                  'European Public Real Estate'], axis=1)
    df.interpolate(method='linear', inplace=True)
    df.dropna(inplace=True)
    df = df.loc['1994-02-28':]

    return df

def load_data_get_ret(path='../../data/price_series.csv'):
    df = pd.read_csv(path, index_col='Time', parse_dates=True)
    df = df.drop(['Hedge Funds Global' ,'Private Equity', 'DAX ',
                  'European Public Real Estate'], axis=1)
    df.interpolate(method='linear', inplace=True)
    df_ret = df.pct_change()
    df_ret.dropna(inplace=True)
    df_ret = df_ret.loc['1994-02-28':]

    return df_ret

def load_data_get_logret(path='../../data/price_series.csv'):
    df = pd.read_csv(path, index_col='Time', parse_dates=True)
    df = df.drop(['Hedge Funds Global', 'Private Equity', 'DAX ',
                  'European Public Real Estate'], axis=1)
    df.interpolate(method='linear', inplace=True)
    df_ret = np.log(df) - np.log(df.shift(1))
    df_ret.dropna(inplace=True)
    df_ret = df_ret.loc['1994-02-28':]  # We start here because otherwise too many assets have monthly values only.

    return df_ret

def get_cov_mat(df_ret):
    return df_ret.cov()


if __name__ == '__main__':
    path = '../data/price_series.csv'
    df = load_data_get_ret(path)
    df_logret = load_data_get_logret(path)