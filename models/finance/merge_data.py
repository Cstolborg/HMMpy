import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../../data/price_series.csv', index_col='Time')
df.index = pd.to_datetime(df.index)

df1 = pd.read_excel('../../data/BarCap_USCorpHY_10yearspread.xlsx')
df1 = df1.iloc[2:]
df1.rename({'ticker': 'Time'}, inplace=True, axis=1)
df1.set_index('Time', inplace=True)
df1.index = pd.to_datetime(df1.index)


df_merged = pd.merge(left=df, right=df1, how='left',
                     left_index=True, right_index=True)

print(df_merged.dropna())