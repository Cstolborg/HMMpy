import copy
import datetime

import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
from scipy import stats
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from statsmodels.tsa.stattools import acf
from utils.data_prep import load_long_series_logret, DataPrep
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM
import warnings
warnings.filterwarnings("ignore")

from analysis.stylized_facts.rolling_acf import compute_rolling_simulations, plot_taylor_effect, plot_acf_data

def plot_acf_subperiods(simulations_subperiods, savefig=None):
    """ Compute absolute acf across 10 subperiods """

    # Compute absolute ACF
    lags = np.arange(simulations_subperiods['mle_acf_sub'].shape[1])

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 10), sharex=True)

    titles = [f'Sup-period {i+1}' for i in range(10)]
    for i, (ax, title) in enumerate(zip(axes.ravel(), titles)):
        ax.set_title(title)
        ax.bar(lags, simulations_subperiods['logrets_acf_sub'][i], label=r'$|r_t|$', color='black', alpha=0.4)
        ax.plot(lags, simulations_subperiods['mle_acf_sub'][i], label="mle")
        ax.plot(lags, simulations_subperiods['jump_acf_sub'][i], label="jump")

        acf_significance = 1.96 / np.sqrt(len(simulations_subperiods['mle'][i]))
        ax.axhline(acf_significance, linestyle='dashed', color='black')
        #ax.set_ylabel(r"ACF($ |r_t|$)")
        ax.set_xlim(left=0, right=101)
        ax.set_ylim(top=0.4, bottom=0)


    plt.legend(fontsize=15)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()




if __name__ == '__main__':
    # Load SP500 logrets
    data = DataPrep()
    logrets = data.load_long_series_logret(outlier_corrected=False)
    logrets_outlier = data.load_long_series_logret(outlier_corrected=True)

    # Instantiate HMM models
    mle = EMHiddenMarkov(n_states=2, epochs=10, max_iter=100, random_state=42)
    jump = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                   epochs=20, max_iter=30, random_state=42)

    #logret = logret[13000:15000]  # Reduce sample size to speed up training

    # Compute dict with long lists of simulations for mle and jump models
    # Also contains acf for both models
    frequency = 1400

    simulations_subperiods = compute_rolling_simulations(logrets, mle, jump, frequency=frequency, window_len=1700,
                                              outlier_corrected=False, n_sims=10000,
                                              compute_acf=False, compute_taylor_effect=True,
                                              compute_acf_subperiods=True)

    # Save results
    save = True
    if save == True:
        plot_acf_subperiods(simulations_subperiods, savefig='acf_abs_subperiods.png')


    else:
        plot_acf_subperiods(simulations_subperiods, savefig=None)
        #plot_taylor_effect(simulations_subperiods, logrets, frequency=frequency, window_len=1700, savefig=None)
        #plot_acf_data(simulations_subperiods, logrets, n_lags=500, savefig=None)
