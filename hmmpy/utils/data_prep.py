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
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
np.seterr(divide='ignore')

@dataclass
class DataPrep:
    """ Class used for loading and preparing data for analysis.

    Primary use case is getting prices, returns and logreturns.
    """
    path: str = '../../data/price_series.csv'
    out_of_sample: bool = True
    drop_insample: List = field(default_factory= lambda: ['Hedge Funds Global', 'Private Equity', 'DAX ',
                  'MSCI World', 'European Public Real Estate'])
    drop_outsample: List = field(default_factory= lambda: ['MSCI World', 'DAX '])
    insample_start: str = '1994-02-28'
    insample_end: str = '2003-11-20'

    def __post_init__(self):
        self.prices = self.load_prices()
        self.rets = self.load_returns()
        self.logrets = self.load_logreturns()
        self.long_logrets = self.load_long_series_logret(outlier_corrected=False)

    def load_prices(self):
        df_prices = pd.read_csv(self.path, index_col='Time', parse_dates=True)

        if self.out_of_sample is True:
            df_prices = df_prices.drop(self.drop_outsample, axis=1)
        else:
            # In-sample data
            df_prices = df_prices.drop(self.drop_insample, axis=1)
            df_prices = df_prices.loc[self.insample_start:self.insample_end]  # Daily data Barclays from 1994 and oos begins 2003

        df_prices.interpolate(method='linear', inplace=True)
        df_prices.dropna(inplace=True)

        return df_prices

    def load_returns(self):
        df_ret = self.prices.pct_change()
        df_ret.dropna(inplace=True)

        return df_ret

    def load_logreturns(self):
        df_ret = np.log(self.prices) - np.log(self.prices.shift(1))
        df_ret.dropna(inplace=True)

        return df_ret

    def load_long_series_logret(self, outlier_corrected=False, threshold=4):
        df = pd.read_csv(self.path, index_col='Time', parse_dates=True)
        df = df['S&P 500 ']
        df_ret = np.log(df) - np.log(df.shift(1))
        df_ret.dropna(inplace=True)

        # Remove trading days with movements above 10%.
        # This includes black monday, two days during GFC, and one day during COVID
        mean, std = df_ret.mean(), df_ret.std()
        df_ret.loc[df_ret >= 0.1] = mean + 6*std
        df_ret.loc[df_ret <= -0.1] = mean - 6*std

        # Remove all observations with std's above 4
        if outlier_corrected is True:
            df_ret = self.replace_outliers(df_ret, threshold=threshold)

        return df_ret

    @staticmethod
    def replace_outliers(data, threshold=4):
        """ Replaces outliers further than x STDs away from mean by the threshold value """
        zscores = stats.zscore(data)
        mean, std = data.mean(), data.std()
        data.loc[zscores >= threshold] = mean + std * threshold
        data.loc[zscores <= -threshold] = mean - std * threshold

        return data

    @staticmethod
    def moving_average(a, n=10):
        a = np.array(a)
        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


######## ALL BELOW FUNCTIONS ARE HERE FOR BACKWARD COMPATIBILITY ONLY. TO BE DEPRECATED #######################

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

def load_returns(path='../../data/price_series.csv', out_of_sample=True):
    df = load_prices(path ,out_of_sample=out_of_sample)
    df_ret = df.pct_change()
    df_ret.dropna(inplace=True)

    return df_ret

def load_logreturns(path='../../data/price_series.csv', out_of_sample=True):
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
    path = '../../data/price_series.csv'
    data = DataPrep(path=path, out_of_sample=True)
    print(data.prices)
    print(data.rets)
    print(data.logrets)

    #data.load_prices()