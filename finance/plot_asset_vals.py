import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/price_series.csv', index_col='Time')
df.dropna(inplace=True)
df = df.iloc[1500:]


port_val = np.load('../data/port_val.npy')

df['port_val'] = port_val
df = df / df.iloc[0] * 100

df.plot()
plt.show()