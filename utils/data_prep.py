import datetime as dt
import numpy as np
import pandas as pd
from scipy import stats

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

def load_prices(path='../../data/price_series.csv', out_of_sample=True):
    df = pd.read_csv(path, index_col='Time', parse_dates=True)

    if out_of_sample is True:
        df = df.drop(['MSCI World', 'DAX '], axis=1)
    else:
        # In-sample data
        df = df.drop(['Hedge Funds Global', 'Private Equity', 'DAX ',
                  'MSCI World', 'European Public Real Estate'] ,axis=1)
        df = df.loc['1994-02-28':'2003-11-20']  # Daily data Barclays from 1994 and oos begins 2003

    df.interpolate(method='linear', inplace=True)
    df.dropna(inplace=True)

    return df

def load_data_get_ret(path='../../data/price_series.csv', out_of_sample=True):
    df = load_prices(path ,out_of_sample=out_of_sample)
    df_ret = df.pct_change()
    df_ret.dropna(inplace=True)

    return df_ret

def load_data_get_logret(path='../../data/price_series.csv', out_of_sample=True):
    df = load_prices(path, out_of_sample=out_of_sample)
    df_ret = np.log(df) - np.log(df.shift(1))
    df_ret.dropna(inplace=True)
    df_ret = df_ret.loc['1994-02-28':]  # We start here because otherwise too many assets have monthly values only.

    return df_ret

def load_long_series_logret(path='../../data/price_series.csv', outlier_corrected=False):
    df = pd.read_csv(path, index_col='Time', parse_dates=True)
    df = df['S&P 500 ']
    df_ret = np.log(df) - np.log(df.shift(1))
    df_ret.dropna(inplace=True)

    # Remove all observations with std's above 4
    if outlier_corrected is True:
        df_ret = df_ret[(np.abs(stats.zscore(df_ret)) < 4)]

    return df_ret

def moving_average(a, n=10) :
    a = np.array(a)
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n





def get_cov_mat(df_ret):
    return df_ret.cov()


if __name__ == '__main__':
    path = '../data/price_series.csv'
    df = load_data_get_ret(path=path)
    df_logret = load_data_get_logret(path=path)