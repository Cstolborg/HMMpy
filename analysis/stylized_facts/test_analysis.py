import copy

import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from utils.data_prep import load_long_series_logret
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM

# Loading data and splitting dataframe
path = '../../analysis/stylized_facts/output_data/'
df = pd.read_csv(path + 'rolling_estimations.csv')

data_table = df.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])

df_mle = df[df['model'] != 'jump']
df_jump = df[df['model'] != 'mle']

# Function for plot MLE estimation
def mle_plot(mu_1 = df_mle['$\mu_1$'], mu_2 = df_mle['$\mu_2$'],
             sigma_1 = df_mle['$\sigma_1$'], sigma_2 = df_mle['$\sigma_2$'],
             q_11 = df_mle['$q_{11}$'], q_22=df_mle['$q_{22}$']):
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

    x_axis = df_mle.index  #Find en måde at erstatte dette på således at vi får årstal.

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
    pass

def jump_plot(mu_1 = df_jump['$\mu_1$'], mu_2 = df_jump['$\mu_2$'],
             sigma_1 = df_jump['$\sigma_1$'], sigma_2 = df_jump['$\sigma_2$'],
             q_11 = df_jump['$q_{11}$'], q_22=df_jump['$q_{22}$']):
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

    x_axis = df_jump.index  #Find en måde at erstatte dette på således at vi får årstal.

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
    pass

if __name__ == '__main__':
    print(df)
    mle_plot()
    jump_plot()



