import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd;
import tqdm
from scipy import stats

pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
from utils.data_prep import load_long_series_logret, moving_average
import warnings
warnings.filterwarnings("ignore")


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
    acf_squared_outlier = sm.tsa.acf(df_returns_outlier**2, nlags=n_lags)[1:]
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
    acf_squared_outlier = sm.tsa.acf(df_returns_outlier**2, nlags=n_lags)[1:]
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
    ax.plot_surface(xx, yy, acf_model_rolling, color='black')

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

    mu_1 = df['$\mu_1$']
    mu_2 = df['$\mu_2$']
    sigma_1 = df['$\sigma_1$']
    sigma_2 = df['$\sigma_2$']
    q_11 = df['$q_{11}$']
    q_22 = df['$q_{22}$']

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharex=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.10)
    ax1 = plt.subplot2grid((3,2),(0,0))
    ax2 = plt.subplot2grid((3,2),(0,1))
    ax3 = plt.subplot2grid((3,2), (1,0))
    ax4 = plt.subplot2grid((3,2),(1,1))
    ax5 = plt.subplot2grid((3,2),(2,0))
    ax6 = plt.subplot2grid((3,2), (2,1))

    # Plotting
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    variables = [mu_1, mu_2, sigma_1, sigma_2, q_11, q_22]
    symbol_list = ['$\mu_1$', '$\mu_2$',
                  '$\sigma_1$', '$\sigma_2$',
                  "$q_{11}$", "$q_{22}$"]

    ## Vi kan indsætte static variables her også, hvis vi vil vise plot hvor 1 model trænes på hele batchen.

    x_axis = df_returns.index[-len(mu_1):]  #Insert the number of trading days in the rolling window.

    for (ax, var, symbol) in zip(axes, variables, symbol_list):
        ax.plot(x_axis, var, color='black')
        ax.set_ylabel(symbol, size=15)
        ax.tick_params('x', labelrotation=45)
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        if ax == ax5 or ax == ax6:
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are on
                top=False,  # ticks along the top edge are off
                labelbottom=True)  # labels along the bottom edge are on

        if symbol == "$q_{11}$" or symbol == "$q_{22}$":
            ax.set_ylim(bottom=0.85, top=1.0)

    if not savefig == None:
        plt.savefig('./images/' + savefig)

    plt.show()

def plot_rolling_moments(df, logrets, window_len=1700, moving_window=50,
                         outlier_corrected=False, savefig=None):
    """ Plot the first four moments of estimated models along with returns"""

    # Refit logreturns into rolling subsamples
    log_rets = get_rolling_logrets(df, logrets,
                                   window_len=1700, moving_window=50, outlier_corrected=outlier_corrected)

    # Compute moments for each subsample of logrets
    empirical_moments = [np.mean(log_rets, axis=1), np.var(log_rets, ddof=1, axis=1),
                         stats.skew(log_rets, axis=1), stats.kurtosis(log_rets, axis=1)]

    # Slice model output data
    df = df.loc[:, ['mean', 'variance', 'skewness', 'excess_kurtosis', 'model']]

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 12), sharex=True)

    models = ['mle', 'jump']
    labels = ['Mean', 'Variance', 'Skewness', 'Excess Kurtosis']
    colors = ['black', 'lightgrey']

    # Loop through each subplot
    for i, (ax, moment, label) in enumerate(zip(axs, empirical_moments, labels)):
        # Plot empirical returns moment
        ax.plot(logrets.index[(window_len):], moment, label=r'$\log (r_t)$', color='grey', ls='--')
        # Inner loop allows drawing of several models
        for (model, color) in zip(models, colors):
            plot_data = df[df['model'] == model]
            ax.plot(plot_data.index[moving_window-1:], moving_average(plot_data.iloc[:, i], n=moving_window),
                    label=model, color=color)


        ax.set_xlim(df.index[moving_window-1], df.index[-1])
        ax.set_ylabel(label)

    axs[-1].tick_params('x', labelrotation=45)
    #axs[1, 1].tick_params('x', labelrotation=45)

    plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)

    plt.show()

if __name__ == '__main__':
    # Load log returns from SP500
    df_returns = load_long_series_logret()
    df_returns_outlier = load_long_series_logret(outlier_corrected=True)

    # Loading regular and outlier corrected data. Then get means parameters across time
    path = '../../analysis/stylized_facts/output_data/'
    df_rolling = pd.read_csv(path + 'rolling_estimations_abs.csv', index_col='timestamp', parse_dates=True)
    df_rolling_outlier = pd.read_csv(path + 'rolling_estimations_outlier_corrected.csv',
                                     index_col='timestamp', parse_dates=True)
    data_table = df_rolling.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])
    data_table_outlier = df_rolling_outlier.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])

    print(data_table)


    save = False
    moving_window = 50
    if save is True:
        plot_rolling_moments(df_rolling, df_returns, moving_window=moving_window, savefig='rolling_moments.png')
        #plot_rolling_moments(df_rolling_outlier, df_returns, moving_window=moving_window, outlier_corrected=True,
        #                     savefig='rolling_moments_outlier_corrected.png')
        plot_acf_2D(data_table, df_returns, savefig='acf_abs_models.png')
        plot_rolling_parameters(df_rolling, model='jump', savefig='2-state JUMP HMM rolling params.png')
        plot_rolling_parameters(df_rolling, model='mle', savefig='2-state MLE HMM rolling params.png')
    else:
        plot_rolling_moments(df_rolling, df_returns, moving_window=moving_window, outlier_corrected=False, savefig=None)
        #plot_rolling_moments(df_rolling_outlier, df_returns, moving_window=moving_window, outlier_corrected=True, savefig=None)
        plot_acf_2D(data_table, df_returns, savefig=None)
        #plot_acf_3D(df_rolling, df_returns, savefig=None)
        plot_rolling_parameters(df_rolling, model='jump', savefig=None)
        plot_rolling_parameters(df_rolling, model='mle', savefig=None)
