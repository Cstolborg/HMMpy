import warnings

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import tqdm
from scipy import stats

from utils.data_prep import load_long_series_logret, moving_average, DataPrep

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)


def get_rolling_logrets(df, logrets, slice=True, window_len=1700, moving_window=50, outlier_corrected=False):
    """Slice log returns into subsamples of window lenghts """
    if slice is True:
        logrets = logrets[-(len(df[df['model'] == 'mle']) + window_len - moving_window):]
    else:
        logrets = logrets[-(len(df) + window_len - moving_window):]

    log_rets = []
    for t in range(window_len, len(logrets)):
        logrets_temp = logrets.iloc[t - window_len:t]

        # If true remove outliers, otherwise do nothing
        if outlier_corrected is True:
            # Outliers more extreme 4 standard deviations are replaced with mean +- 4*std.
            outlier_value_pos = logrets_temp.mean() + 4 * logrets_temp.std()
            outlier_value_neg = logrets_temp.mean() - 4 * logrets_temp.std()

            # Handle positive and negative returns separately
            outliers_idx = (np.abs(stats.zscore(logrets_temp)) >= 4)
            logrets_temp[outliers_idx & (logrets_temp > 0)] = outlier_value_pos
            logrets_temp[outliers_idx & (logrets_temp < 0)] = outlier_value_neg

            log_rets.append(logrets_temp)
        else:
            log_rets.append(logrets_temp)

    return log_rets

# Function to plot the empirical ACF squared and the simulated solution.
def plot_acf_2D(data_table, df_returns, savefig=None):
    n_lags = len(data_table.loc[:, 'lag_0':].columns)
    lags = [i for i in range(n_lags)]
    acf_squared = sm.tsa.acf(df_returns**2, nlags=n_lags)[1:]
    acf_squared_outlier = sm.tsa.acf(logrets_outlier ** 2, nlags=n_lags)[1:]
    acf_square_conf = 1.96 / np.sqrt(len(df_returns))

    squared_acf_SP500_mle = data_table.loc[(1700, 'mle'), 'lag_0':]
    squared_acf_SP500_jump = data_table.loc[(1700, 'jump'), 'lag_0':]

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 7), sharex=True)

    # Full data
    ax[0].set_title('Full sample')
    ax[0].bar(lags, acf_squared, color='black', alpha=0.4)
    ax[0].plot(lags, squared_acf_SP500_mle, label="mle")
    ax[0].plot(lags, squared_acf_SP500_jump, label="jump")

    # Outlier-corrected
    ax[1].set_title(r'Outliers limited to $\bar r_t \pm 4\sigma$')
    ax[1].bar(lags, acf_squared_outlier, color='black', alpha=0.4)
    ax[1].plot(lags, data_table_outlier.loc[(1700, 'mle'), 'lag_0':], label="mle")
    ax[1].plot(lags, data_table_outlier.loc[(1700, 'jump'), 'lag_0':], label="jump")
    ax[1].set_xlabel('Lag')

    for i in range(len(ax)):
        ax[i].axhline(acf_square_conf, linestyle='dashed', color='black')
        ax[i].set_ylabel("ACF squared(log $r_t)$")
        ax[i].set_xlim(left=0, right=max(lags))
        ax[i].set_ylim(top=0.4, bottom=0)

    plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

# Function to plot the empirical ACF squared and the simulated solution.
def plot_acf_3D(df, logrets, model='mle', savefig=None):
    # Slice df and logrets so shapes match
    df = df[df['model'] == model]
    df = df.loc[:, 'lag_0':'lag_99']
    #logrets = logrets.iloc[-len(df):]

    n_lags = len(data_table.loc[:, 'lag_0':].columns)
    lags = np.arange(n_lags)

    log_rets = get_rolling_logrets(df, logrets, slice=False, window_len=1700, moving_window=0, outlier_corrected=False)
    log_rets_squared = np.array(log_rets) ** 2

    acf_rolling = []
    acf_model_rolling = []
    for t in tqdm.tqdm(range(0, len(log_rets_squared), 250)):
        # Get n_lags for each time period for log returns
        acf_squared = sm.tsa.acf(log_rets_squared[t], nlags=n_lags)[1:]
        acf_rolling.append(acf_squared)

        # Do the same for the model - note already calculated in data
        acf_model_rolling.append(df.iloc[t])

    acf_rolling = np.array(acf_rolling)
    acf_model_rolling = np.array(acf_model_rolling)

    #acf_squared = sm.tsa.acf(logrets ** 2, nlags=n_lags)[1:]
    acf_squared_outlier = sm.tsa.acf(logrets_outlier ** 2, nlags=n_lags)[1:]
    acf_square_conf = 1.96 / np.sqrt(len(logrets))

    squared_acf_SP500_mle = data_table.loc[(1700, 'mle'), 'lag_0':]
    squared_acf_SP500_jump = data_table.loc[(1700, 'jump'), 'lag_0':]

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data as bar chart
    time = np.arange(acf_rolling.shape[0])
    xx, yy = np.meshgrid(lags, time)
    x, y = xx.ravel(), yy.ravel()
    z = np.zeros(len(x))

    dx = np.ones(len(x))
    dy = np.ones(len(x))
    dz = acf_rolling.ravel()
    #ax.bar3d(x, y, z, dx, dy, dz, shade=True, color='lightgrey', alpha=0.2)


    # Plot model as surface
    ax.plot_surface(xx, yy, acf_rolling, color='lightgrey', alpha=0.2)
    ax.plot_surface(xx, yy, acf_model_rolling)

    ax.set_xlabel('lag')
    ax.set_ylabel('Time step')
    ax.set_zlabel('ACF')

    ax.set_zlim(top=0.4, bottom=0)

    plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

def plot_rolling_parameters(df, model ='mle', savefig=None):
    """ Function for plotting rolling parameters for different estimation procedures. """
    df = df[df['model'] == model]

    # Plotting
    plt.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(3,2, figsize=(15,12), sharex=True)

    symbol_list = ['$\mu_1$', '$\mu_2$',
                  '$\sigma_1$', '$\sigma_2$',
                  "$q_{11}$", "$q_{22}$"]

    for (ax, symbol) in zip(axes.flatten(), symbol_list):
        ax.plot(df[symbol], color='black')
        ax.set_ylabel(symbol)
        ax.tick_params('x', labelrotation=45)

        if symbol == "$q_{11}$" or symbol == "$q_{22}$":
            ax.set_ylim(bottom=0.85, top=1.0)

    plt.tight_layout()
    if not savefig == None:
        plt.savefig('./images/' + savefig)

    plt.show()


def plot_moments_regular(df, logrets, window_len=1700, moving_window=50,
                         outlier_corrected=False, savefig=None):
    """ Plot the first four moments of estimated models along with returns"""

    # Refit logreturns into rolling subsamples
    log_rets = get_rolling_logrets(df, logrets,
                                   window_len=1700, moving_window=moving_window, outlier_corrected=outlier_corrected)

    # Compute moments for each subsample of logrets
    empirical_moments = [np.mean(log_rets, axis=1), np.var(log_rets, ddof=1, axis=1),
                         stats.skew(log_rets, axis=1), stats.kurtosis(log_rets, axis=1)]

    # Slice model output data
    df = df.loc[:, ['mean', 'variance', 'skewness', 'excess_kurtosis', 'model']]

    # Plotting
    plt.rcParams.update({'font.size': 25})
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 12), sharex=True)

    models = ['mle', 'jump']
    labels = ['Mean', 'Variance', 'Skewness', 'Excess Kurtosis']

    # Loop through each subplot
    for i, (ax, moment, label) in enumerate(zip(axs, empirical_moments, labels)):
        # Plot empirical returns moment
        ax.plot(logrets.index[(window_len+moving_window):], moment, label=r'$(r_t)$', color='grey', ls='--')
        # Inner loop allows drawing of several models
        for model in models:
            plot_data = df[df['model'] == model]
            ax.plot(plot_data.index[moving_window-1:], moving_average(plot_data.iloc[:, i], n=moving_window),
                    label=model)


        ax.set_xlim(df.index[moving_window-1], df.index[-1])
        ax.set_ylabel(label)

    axs[-1].tick_params('x', labelrotation=45)
    #axs[1, 1].tick_params('x', labelrotation=45)

    plt.legend(fontsize=15)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)

    plt.show()

def plot_moments_bulla(df, logrets, window_len=1700, moving_window=50,
                       outlier_corrected=False, savefig=None):
    """ Plot the first four moments of estimated models along with returns"""

    # Refit logreturns into rolling subsamples
    log_rets = get_rolling_logrets(df, logrets,
                                   window_len=1700, moving_window=moving_window, outlier_corrected=outlier_corrected)

    # Compute moments for each subsample of logrets
    empirical_moments = [np.mean(log_rets, axis=1) / np.std(log_rets, ddof=1, axis=1),
                         stats.skew(log_rets, axis=1), stats.kurtosis(log_rets, axis=1)]

    # Slice model output data
    df['mean/std'] = df['mean'] / np.sqrt(df['variance'])
    df = df.loc[:, ['mean/std', 'skewness', 'excess_kurtosis', 'model']]

    # Plotting
    plt.rcParams.update({'font.size': 25})
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True)

    models = ['mle', 'jump']
    labels = ['Mean/Std', 'Skewness', 'Excess Kurtosis']

    # Loop through each subplot
    for i, (ax, moment, label) in enumerate(zip(axes, empirical_moments, labels)):
        # Plot empirical returns moment
        ax.plot(logrets.index[(window_len+moving_window):], moment, label=r'$(r_t)$', color='grey', ls='--')
        # Inner loop allows drawing of several models
        for model in models:
            plot_data = df[df['model'] == model]
            ax.plot(plot_data.index[moving_window-1:], DataPrep.moving_average(plot_data.iloc[:, i], n=moving_window),
                    label=model)


        ax.set_xlim(df.index[moving_window-1], df.index[-1])
        ax.set_ylabel(label)

    axes[-1].tick_params('x', labelrotation=45)

    plt.legend(fontsize=15)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)

    plt.show()

if __name__ == '__main__':
    # Load log returns from SP500
    data = DataPrep()
    logrets = data.load_long_series_logret(outlier_corrected=False)
    logrets_outlier = data.load_long_series_logret(outlier_corrected=True)

    # Loading regular and outlier corrected data. Then get means parameters across time
    path = '../../analysis/stylized_facts/output_data/'
    df = pd.read_csv(path + 'moments_abs.csv', index_col='timestamp', parse_dates=True)
    df_outlier = pd.read_csv(path + 'moments_abs_outlier.csv',
                             index_col='timestamp', parse_dates=True)
    data_table = df.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])
    data_table_outlier = df_outlier.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])

    print(data_table)

    save = False
    moving_window = 50
    if save is True:
        #Regular data
        plot_moments_bulla(df, np.abs(logrets),
                           moving_window=moving_window, outlier_corrected=False,
                           savefig='moments_bulla_abs.png')
        plot_moments_regular(df, np.abs(logrets), moving_window=moving_window, savefig='moments_regular_abs.png')

        plot_rolling_parameters(df, model='jump', savefig='2-state JUMP HMM rolling params.png')
        plot_rolling_parameters(df, model='mle', savefig='2-state MLE HMM rolling params.png')

        #Outlier corrected
        plot_moments_bulla(df_outlier, np.abs(logrets_outlier),
                           moving_window=moving_window, outlier_corrected=False,
                           savefig='moments_bulla_abs_outlier.png')
    else:
        # Regular data
        plot_moments_bulla(df, np.abs(logrets), moving_window=moving_window, outlier_corrected=False, savefig=None)
        plot_moments_bulla(df_outlier, np.abs(logrets_outlier), moving_window=moving_window, outlier_corrected=False, savefig=None)

        plot_moments_regular(df, logrets, moving_window=moving_window, outlier_corrected=False, savefig=None)

        plot_rolling_parameters(df, model='jump', savefig=None)
        plot_rolling_parameters(df, model='mle', savefig=None)

        # Outlier corrected
        plot_moments_bulla(df_outlier, logrets_outlier, moving_window=moving_window, outlier_corrected=False, savefig=None)
        plot_moments_regular(df_outlier, logrets_outlier, moving_window=moving_window, savefig=None)

        # Not used plots
        #plot_acf_3D(df, logrets, savefig=None)