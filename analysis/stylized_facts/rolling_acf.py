import copy

import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
from scipy import stats
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from statsmodels.tsa.stattools import acf
from utils.data_prep import load_long_series_logret
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM
import warnings
warnings.filterwarnings("ignore")


def compute_rolling_simulations(logret, mle, jump, frequency=100, window_len=1700,
                                n_lags=500, n_sims=5000, outlier_corrected=False,
                                compute_taylor_effect=True, compute_acf=True,
                                compute_acf_subperiods=False):
    n_obs = len(logret)

    simulations = {'mle': [],
                   'jump': [],
                   'mle_taylor': np.array([]),
                   'jump_taylor': np.array([]),
                   'logrets_taylor': np.array([]),
                   'logrets_acf_sub': np.array([]),
                   'mle_acf_sub': np.array([]),
                   'jump_acf_sub': np.array([])}

    # Loop through data and fit models at each time step
    for t in tqdm.tqdm(range(window_len, n_obs, frequency)):
        # Slice data into rolling sequences
        rolling = logret.iloc[t - window_len: t]

        # Remove all observations with std's above 4
        if outlier_corrected is True:
            rolling = rolling[(np.abs(stats.zscore(rolling)) < 4)]

        # Fit models to rolling data
        mle.fit(rolling, sort_state_seq=True, verbose=True)
        jump.fit(rolling, sort_state_seq=True, get_hmm_params=True, verbose=True)

        ## Simulate data for ACF
        mle_simulation = mle.sample(n_samples=n_sims)[0]  # Simulate returns
        jump_simulation = jump.sample(n_samples=n_sims)[0]

        simulations['mle'].append(mle_simulation)
        simulations['jump'].append(jump_simulation)

        if compute_taylor_effect == True:
            # Maximize first-order acf for rolling logrets and models
            object_fun = lambda power, data: -acf(np.abs(data) ** power, nlags=1)[-1]
            simulations['logrets_taylor'] = np.append(simulations['logrets_taylor'],
                                                      opt.minimize(object_fun, x0=1, args=rolling).x)
            simulations['mle_taylor'] = np.append(simulations['mle_taylor'],
                                                  opt.minimize(object_fun, x0=1, args=simulations['mle'][-1]).x)
            simulations['jump_taylor'] = np.append(simulations['jump_taylor'],
                                                   opt.minimize(object_fun, x0=1, args=simulations['jump'][-1]).x)

        if compute_acf_subperiods == True:
            simulations['logrets_acf_sub'] = np.append(simulations['logrets_acf_sub'], acf(np.abs(rolling), nlags=n_lags)[1:])
            simulations['mle_acf_sub'] = np.append(simulations['mle_acf_sub'],
                                                   acf(np.abs(simulations['mle'][-1]), nlags=n_lags)[1:])
            simulations['jump_acf_sub'] = np.append(simulations['jump_acf_sub'],
                                                    acf(np.abs(simulations['jump'][-1]), nlags=n_lags)[1:])
    # Reshape subperiods
    if compute_acf_subperiods == True:
        simulations['logrets_acf_sub'] = simulations['logrets_acf_sub'].reshape(-1, n_lags)
        simulations['mle_acf_sub'] = simulations['mle_acf_sub'].reshape(-1, n_lags)
        simulations['jump_acf_sub'] = simulations['jump_acf_sub'].reshape(-1, n_lags)

    simulations['mle'] = np.array(simulations['mle'])
    simulations['jump'] = np.array(simulations['jump'])

    if compute_acf == True:
        print('Computing ACF on simulated data...')
        simulations['mle_acf'] = acf(np.abs(simulations['mle'][:, :2000].ravel()), nlags=n_lags)[1:]
        simulations['jump_acf'] = acf(np.abs(simulations['jump'][:, :2000].ravel()), nlags=n_lags)[1:]
        print('Finished computing ACF')

    return simulations

def plot_simulated_acf(simulations, logret,n_lags=500, savefig=None):
    """ Compute absolute or squared acf in the long-run """
    # Compute absolute ACF
    acf_logret = acf(np.abs(logret), nlags=n_lags)[1:]
    acf_significance = 1.96 / np.sqrt(len(logret))

    lags = np.arange(simulations['mle_acf'].size)

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharex=True)

    # ACF of data and models
    ax.bar(lags, acf_logret, color='black', alpha=0.4)
    ax.plot(lags, simulations['mle_acf'].ravel(), label="mle", color='lightgrey')
    ax.plot(lags, simulations['jump_acf'].ravel(), label="jump", color='black')

    ax.axhline(acf_significance, linestyle='dashed', color='black')
    ax.set_ylabel(r"ACF($\log |r_t|$)")
    ax.set_xlim(left=0, right=max(lags)+1)
    ax.set_ylim(top=0.4, bottom=0)

    plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

def plot_simulated_acf_outliers(simulations, simulations_outliers, logret, logret_outliers,
                                n_lags=500, savefig=None):
    """ Compute absolute or squared acf in the long-run """
    # Compute absolute ACF
    acf_logret = acf(np.abs(logret), nlags=n_lags)[1:]
    acf_logret_outliers = acf(np.abs(logret_outliers), nlags=n_lags)[1:]
    acf_significance = 1.96 / np.sqrt(len(logret))
    lags = np.arange(simulations['mle_acf'].size)

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 7), sharex=True)

    # Full data
    ax[0].set_title('Full sample')
    ax[0].bar(lags, acf_logret, color='black', alpha=0.4)
    ax[0].plot(lags, simulations['mle_acf'].ravel(), label="mle", color='lightgrey')
    ax[0].plot(lags, simulations['jump_acf'].ravel(), label="jump", color='black')

    # Outlier-corrected
    ax[1].set_title(r'Outliers limited to $\bar r_t \pm 4\sigma$')
    ax[1].bar(lags, acf_logret_outliers, color='black', alpha=0.4)
    ax[1].plot(lags, simulations_outliers['mle_acf'].ravel(), label="mle", color='lightgrey')
    ax[1].plot(lags, simulations_outliers['jump_acf'].ravel(), label="jump", color='black')
    ax[1].set_xlabel('Lag')

    for i in range(len(ax)):
        ax[i].axhline(acf_significance, linestyle='dashed', color='black')
        ax[i].set_ylabel(r"ACF($\log |r_t|$)")
        ax[i].set_xlim(left=0, right=max(lags)+1)
        ax[i].set_ylim(top=0.4, bottom=0)

    plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()




if __name__ == '__main__':
    # Load SP500 logrets
    logret = load_long_series_logret()
    logret_outliers = load_long_series_logret(outlier_corrected=True)

    # Instantiate HMM models
    mle = EMHiddenMarkov(n_states=2, epochs=10, max_iter=100, random_state=42)
    jump = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14),
                   epochs=20, max_iter=30, random_state=42)

    #logret = logret[13000:15000]  # Reduce sample size to speed up training

    # Compute dict with long lists of simulations for mle and jump models
    # Also contains acf for both models
    frequency = 100
    simulations = compute_rolling_simulations(logret, mle, jump, frequency=frequency, window_len=1700,
                                              outlier_corrected=False, n_sims=10000,
                                              compute_acf=True, compute_taylor_effect=True)


    # Repeat procedure with outlier corrected data
    #simulations_outliers = compute_rolling_simulations(logret, mle, jump, frequency=100, window_len=1700,
    #                                          outlier_corrected=False, n_sims=1700)





    # Save results
    save = False
    if save == True:
        plot_simulated_acf(simulations, logret, n_lags=500, savefig='simulated_abs_acf.png')
        #plot_simulated_acf_outliers(simulations, simulations_outliers, logret, logret_outliers,
        #                            n_lags=500, savefig='simulated_abs_acf_outliers.png')
    else:
        plot_simulated_acf(simulations, logret, n_lags=500, savefig=None)
        #plot_simulated_acf_outliers(simulations, simulations_outliers, logret, logret_outliers,
        #                            n_lags=500, savefig=None)

