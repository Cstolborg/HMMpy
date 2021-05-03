import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('../../data/price_series_old.csv', index_col='Time')
    df.index = pd.to_datetime(df.index)

    df1 = pd.read_excel('../../data/BarCap_USCorpHY_10yearspread.xlsx')
    df1 = df1.iloc[2:]
    df1.rename({'ticker': 'Time'}, inplace=True, axis=1)
    df1.set_index('Time', inplace=True)
    df1.index = pd.to_datetime(df1.index)


    df_merged = pd.merge(left=df, right=df1, how='left',
                         left_index=True, right_index=True)

    # Rearrange columns so rf is the last column
    cols = ['Hedge Funds Global',
             'MSCI World',
             'MSCI Emerging Markets',
             'DAX ',
             'European Public Real Estate',
             'S&P 500 ',
             'Oil',
             'Gold index',
             'Barclays US Treasury',
             'Private Equity',
             'CSI BARC Index',
             'T-bills rf']

    df_merged = df_merged[cols]
    df_merged.to_csv('../../data/price_series.csv')

    print(df_merged)