import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from analysis.portfolio_exercise.mean_var import mean_var
from analysis.portfolio_exercise.data_description import compute_asset_metrics
from hmmpy.finance.backtest import Backtester
from hmmpy.finance.backtest import FinanceHMM
from hmmpy.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from hmmpy.hidden_markov.hmm_jump import JumpHMM
from hmmpy.utils.data_prep import DataPrep

np.seterr(divide='ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

def merge_data():
    df1 = pd.read_csv('output_data/mle/frontiers_lo.csv')
    df1 = df1[(df1['short_cons'] == 'LO') & (df1['D_max'] >= 0.15)]

    df = df1.drop(columns=['gamma_1', 'gamma_3', 'gamma_10', 'gamma_15', 'gamma_25'])
    df = df.pivot(index='timestamp', columns='D_max', values='gamma_5')

    df = df.rename(columns={0.15: '$LO_{D_{max}=0.15}$', 1000.00: 'LO'})

    df1 = pd.read_csv('output_data/mle/archive/frontiers_ls.csv')
    df1 = df1[(df1['short_cons'] == 'LS') & (df1['D_max'] >= 0.15)]

    df2 = df1.drop(columns=['gamma_1', 'gamma_3', 'gamma_10', 'gamma_15', 'gamma_25'])
    df2 = df2.pivot(index='timestamp', columns='D_max', values='gamma_5')

    df['$LS_{D_{max}=0.15}$'] = df2[0.15]
    df['LS'] = df2[1000]

    equal_weigthed = Backtester()

    w_tan = mean_var(data)
    equal_weigthed.backtest_equal_weighted(data.rets.iloc[1000:], use_weights=w_tan)
    df['FM'] = equal_weigthed.port_val

    equal_weigthed.backtest_equal_weighted(data.rets.iloc[1000:], rebal_freq='M')
    df['1/n'] = equal_weigthed.port_val

    df.index = pd.to_datetime(df.index)
    df['T-bills rf'] = data.prices.iloc[1000:]['T-bills rf']

    return df

def plot_port_val(df, savefig=None):
    df = df.drop(columns='T-bills rf')

    # Compute drawdowns
    peaks = df.cummax(axis=0)
    drawdown = (df - peaks) / peaks

    # Plotting
    plt.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2, colspan=1)
    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1, sharex=ax1)

    #for col in df_frontiers.groupby(['short_cons', 'D_max']):
    #    label = f'${type[0]}_{{D_{{max}}={type[1]}}}$' if type[1] < 1. else f'${type[0]}$'
    #    ax.plot(data['timestamp'], data['gamma_5'], label=label)

    df.plot(ax=ax1)
    drawdown.iloc[:, [0,2,5]].plot(ax=ax2)

    ax1.set_ylabel('$P_t$')

    ax2.set_ylabel('Drawdown')
    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()])
    ax2.tick_params('x', labelrotation=45)
    ax2.set_xlabel('')
    ax2.legend(fontsize=15)


    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()


if __name__ == '__main__':
    data = DataPrep(out_of_sample=True)

    df = pd.read_csv('./output_data/comparison.csv', index_col='timestamp', parse_dates=True)
    #df = merge_data()
    #df.to_csv('./output_data/comparison.csv')

    metrics = compute_asset_metrics(df)
    metrics.to_latex('./output_data/port_performance.tex')

    plot_port_val(df, savefig='comparison_perf.png')