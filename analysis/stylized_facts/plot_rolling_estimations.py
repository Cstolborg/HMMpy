import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

from utils.data_prep import load_long_series_logret

import warnings
warnings.filterwarnings("ignore")

# Load log returns from SP500
df_returns = load_long_series_logret()

# Loading data for fitted models
path = '../../analysis/stylized_facts/output_data/'
df = pd.read_csv(path + 'rolling_estimations.csv')
data_table = df.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])

# Loading returns data
path_1 = '../../data/'
df_returns = pd.read_csv(path_1 + 'price_series.csv', index_col = 'Time')
df_returns.index = pd.to_datetime(df_returns.index)

# Splitting the data from df into jump and MLE
df_mle = df[df['model'] == 'mle']
df_jump = df[df['model'] == 'jump']

# Function to compute returns from the data of the S&P 500
df_SP500 = df_returns[['S&P 500 ']]
df_SP500['S&P 500 Index'] = df_SP500['S&P 500 '] / df_SP500['S&P 500 '][0] * 100
df_SP500['Returns'] = df_SP500['S&P 500 Index'].pct_change()
df_SP500['Log returns'] = np.log(df_SP500['S&P 500 Index']) - np.log(df_SP500['S&P 500 Index'].shift(1))

# Get squared ACF from data_table
n_lags = len(data_table.loc[:, 'lag_0':].columns)
squared_acf_SP500_mle = data_table.loc[(1700, 'mle'), 'lag_0':]
squared_acf_SP500_jump = data_table.loc[(1700, 'jump'), 'lag_0':]

# Function to compute and plot acf and squared acf of the S&P 500 log returns
def acfsquared_SP500_mle():  # TODO to be deleted
    # Derive  Squared ACF
    lags = [i for i in range(n_lags)]
    acf_squared = sm.tsa.acf(np.square(df_SP500['Log returns'].dropna()),nlags = n_lags)[1:]

    # Confidence interval for ACF^2
    acf_square_conf = [1.96 / np.sqrt(len(np.square(df_SP500['Log returns'].dropna()))),
                    - 1.96 / np.sqrt(len(np.square(df_SP500['Log returns'].dropna())))]

    # Plot squared ACF with generated squared
    fig, ax2 = plt.subplots(figsize=(15,7))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    ax2.bar(lags, acf_squared, color='black', alpha=0.4)
    ax2.axhline(acf_square_conf[0], linestyle='dashed', color='black')
    ax2.axhline(acf_square_conf[1], linestyle='dashed', color='black')
    ax2.set_ylabel("ACF squared(log $r_t)$")
    ax2.set_xlabel('Lag')
    ax2.set_xlim(left=0.5, right=max(lags)+1)
    plt.tight_layout()
    plt.show()

# Function to plot the empirical ACF squared and the simulated solution.
def plot_acfsquared_SP500_combined():
    lags = [i for i in range(n_lags)]
    acf_squared = sm.tsa.acf(np.square(df_SP500['Log returns'].dropna()), nlags=n_lags)[1:]
    acf_square_conf = 1.96 / np.sqrt(len(np.square(df_SP500['Log returns'].dropna())))

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12), sharex=True)

    for i in range(len(ax)):
        ax[i].bar(lags, acf_squared, color='black', alpha=0.4)
        ax[i].plot(lags, squared_acf_SP500_mle, label="mle")
        ax[i].plot(lags, squared_acf_SP500_jump, label="jump")
        ax[i].axhline(acf_square_conf, linestyle='dashed', color='black')
        ax[i].set_ylabel("ACF squared(log $r_t)$")
        ax[i].set_xlabel('Lag')
        ax[i].set_xlim(left=0.5, right=max(lags) + 1)

    plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()
    plt.show()

# Function for plot MLE estimation
def mle_plot(mu_1 = df_mle['$\mu_1$'], mu_2 = df_mle['$\mu_2$'],
             sigma_1 = df_mle['$\sigma_1$'], sigma_2 = df_mle['$\sigma_2$'],
             q_11 = df_mle['$q_{11}$'], q_22=df_mle['$q_{22}$']):
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
        ax.plot(x_axis, var)
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

    plt.savefig("2-state MLE HMM rolling params")

    plt.show()

def jump_plot(mu_1 = df_jump['$\mu_1$'], mu_2 = df_jump['$\mu_2$'],
             sigma_1 = df_jump['$\sigma_1$'], sigma_2 = df_jump['$\sigma_2$'],
             q_11 = df_jump['$q_{11}$'], q_22=df_jump['$q_{22}$']):
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

    x_axis = df_returns.index[-len(mu_1):]  # Insert the number of rolling trading days

    for (ax, var, symbol) in zip(axes, variables, symbol_list):
        ax.plot(x_axis, var)
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

    plt.savefig("2-state JUMP HMM rolling params")

    plt.show()

if __name__ == '__main__':
    print(data_table)
    #print(data_table.columns.values)

    #acfsquared_SP500_mle()
    plot_acfsquared_SP500_combined()
    #mle_plot()
    #jump_plot()









