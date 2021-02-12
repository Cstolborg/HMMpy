import numpy as np
import pandas as pd


def load_data_get_ret(path='../data/adjusted_close_price_series_load.csv'):
    df = pd.read_csv(path, index_col='Time / Name')
    df.dropna(inplace=True)

    df_ret = df[['Hedge Funds Global', 'MSCI World', 'MSCI Emerging Markets',
                 'Barclays US Treasury Bond Index', 'S&P Listed Private Equity Index',
                 'European Public Real Estate', 'S&P Crude Oil Index', 'Gold']].pct_change()

    df_ret.dropna(inplace=True)

    return df_ret

def get_cov_mat(df_ret):
    return df_ret.cov()
