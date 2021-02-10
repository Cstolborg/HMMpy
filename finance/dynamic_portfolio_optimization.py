import numpy as np
import pandas as pd
from utils import hmm_sampler

# Use in full-scale model. 
#model = hmm_sampler.SampleHMM(n_states=2)
#n_samples = 10
#n_sequences = 1
#X, viterbi_states, true_states = model.sample_with_viterbi(n_samples, n_sequences)


df = pd.read_excel('../data/adjusted_close_price_series_load.xlsx', header = 2, index_col = 'Time / Name')
pd.set_option('display.max_columns', None)
df.dropna(inplace=True)

#Derive returns of the price series
df_returns = df[['Hedge Funds Global', 'MSCI World', 'MSCI Emerging Markets',
            'Barclays US Treasury Bond Index', 'S&P Listed Private Equity Index',
            'European Public Real Estate','S&P Crude Oil Index','Gold']].pct_change()
df_returns.dropna(inplace=True)

df_returns = df_returns.iloc[-10:] ## - Dummy for 10 periods would have to be the entire series for full-scale setup


## From here code for h_t
state_sequence = [0,0,1,0,1,0,1,1,1,1] ## TODO - We have to forecast mean and variance of the assets based on the HMM... This is purely dummy stuff that serves as nice to have for the setup

Weight_asset = np.zeros(shape=(len(state_sequence), 8))  # Init as empty matrix to store the weight in a specific asset.

for i in range(len(state_sequence)):
    if state_sequence[i] == 0:
        Weight_asset[i] = ([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])
    elif state_sequence[i] == 1:
        Weight_asset[i] = ([0, 0.30, 0, 0.25, 0, 0.15, 0, 0.3])

Weight_asset = pd.DataFrame(Weight_asset, columns=['Hedge Funds Global', 'MSCI World', 'MSCI Emerging Markets',
            'Barclays US Treasury Bond Index', 'S&P Listed Private Equity Index',
            'European Public Real Estate','S&P Crude Oil Index','Gold'])


port_val = np.zeros(shape=(df_returns.shape[0], df_returns.shape[1]))
port_val[0] = Weight_asset.iloc[0, :] * 1000 * (1 + df_returns.iloc[0, :])



for i in range(1, len(df_returns)):
    port_val[i] = port_val[i-1] * (1 + df_returns.iloc[i, :])
    port_val_total = np.sum(port_val, axis =1)
