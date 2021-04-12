import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
from utils.data_prep import load_long_series_logret
import warnings
warnings.filterwarnings("ignore")

# Load log returns from SP500
df_returns = load_long_series_logret()
df_returns_outlier = load_long_series_logret(outlier_corrected=True)

# Loading regular and outlier corrected data. Then get means parameters across time
path = '../../analysis/stylized_facts/output_data/'
df = pd.read_csv(path + 'rolling_estimations.csv')
df_outlier = pd.read_csv(path + 'rolling_estimations_outlier_corrected.csv')
data_table = df.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])
data_table_outlier = df_outlier.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])

# Splitting the data from df into jump and MLE
df_mle = df[df['model'] == 'mle']
df_jump = df[df['model'] == 'jump']

# Function to plot the empirical ACF squared and the simulated solution.
def plot_acfsquared_SP500_combined(data_table):
    n_lags = len(data_table.loc[:, 'lag_0':].columns)
    lags = [i for i in range(n_lags)]
    acf_squared = sm.tsa.acf(np.square(df_returns), nlags=n_lags)[1:]
    acf_squared_outlier = sm.tsa.acf(df_returns_outlier**2, nlags=n_lags)[1:]
    acf_square_conf = 1.96 / np.sqrt(len(np.square(df_returns)))

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
    ax[1].set_title(r'Outliers limited to $\bar r_t \pm 0$')
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
    plt.show()

# Function for plotting rolling parameters for different estimation procedures.
def plot_rolling_parameters(plot_type = 'mle'):
    if plot_type == 'mle':
        mu_1 = df_mle['$\mu_1$']
        mu_2 = df_mle['$\mu_2$']
        sigma_1 = df_mle['$\sigma_1$']
        sigma_2 = df_mle['$\sigma_2$']
        q_11 = df_mle['$q_{11}$']
        q_22 = df_mle['$q_{22}$']

    elif plot_type == 'jump':
        mu_1 = df_jump['$\mu_1$']
        mu_2 = df_jump['$\mu_2$']
        sigma_1 = df_jump['$\sigma_1$']
        sigma_2 = df_jump['$\sigma_2$']
        q_11 = df_jump['$q_{11}$']
        q_22 = df_jump['$q_{22}$']

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


if __name__ == '__main__':
    print(data_table)
    #print(data_table.columns.values)

    #acfsquared_SP500_mle()
    plot_acfsquared_SP500_combined(data_table)
    #plot_rolling_parameters(plot_type='jump')










