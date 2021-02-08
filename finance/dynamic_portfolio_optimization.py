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

df_returns = df_returns.iloc[-10:] ## - Dummy for 10 periods would have to be the entire series for full-scale setup

state_sequence = [0,0,1,0,1,0,1,1,1,1] ## TODO - We have to forecast mean and variance of the assets based on the HMM... This is purely dummy stuff that serves as nice to have for the setup

Weight_asset = np.zeros(shape=(len(state_sequence), 8))  # Init as empty matrix to store the weight in a specific asset.

Capital_t0 = np.array([1000,1000,1000,1000,1000,1000,1000,1000])

for i in range(len(state_sequence)):
    if state_sequence[i] == 0:
        Weight_asset[i] = ([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])
    elif state_sequence[i] == 1:
        Weight_asset[i] = ([0, 0.30, 0, 0.25, 0, 0.15, 0, 0.3])

Weight_asset = pd.DataFrame(Weight_asset, columns=['Hedge Funds Global', 'MSCI World', 'MSCI Emerging Markets',
            'Barclays US Treasury Bond Index', 'S&P Listed Private Equity Index',
            'European Public Real Estate','S&P Crude Oil Index','Gold'])

Portfolio_t0 = Weight_asset.iloc[0,:]*Capital_t0

print(Portfolio_t0)








