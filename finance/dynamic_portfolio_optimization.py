import numpy as np
import pandas as pd

df = pd.read_excel('../data/adjusted_close_price_series_load.xlsx', header = 2, index_col = 'Time / Name')
pd.set_option('display.max_columns', None)
df.dropna(inplace=True)

#Derive returns of the price series
df_returns = df[['Hedge Funds Global', 'MSCI World', 'MSCI Emerging Markets',
            'Barclays US Treasury Bond Index', 'S&P Listed Private Equity Index',
            'European Public Real Estate','S&P Crude Oil Index','Gold']].pct_change()
df_returns.dropna(inplace=True)

state_sequence = [0,0,1,0,1,0,1,1,1,1]

Weight_asset = np.zeros(shape=(len(state_sequence), 8))  # Init as empty matrix

for i in range(len(state_sequence)):
    if state_sequence == 0:
        Weight_asset = ([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])
    elif state_sequence == 1:
        Weight_asset = ([0, 0.30, 0, 0.25, 0, 0.15, 0, 0.3])


print(Weight_asset)


