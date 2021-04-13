import copy

import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from statsmodels.tsa.stattools import acf
from utils.data_prep import load_long_series_logret
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM
import warnings
warnings.filterwarnings("ignore")


def rolling_state_decoding(logret, mle, jump, window_len=1700,
                                outlier_corrected=False):
    n_obs = len(logret)

    decoded_states = {'mle': [],
                   'jump': []}

    # Loop through data and fit models at each time step
    for t in tqdm.tqdm(range(window_len, n_obs)):
        # Slice data into rolling sequences
        rolling = logret.iloc[t - window_len: t]

        # Remove all observations with std's above 4
        if outlier_corrected is True:
            rolling = rolling[(np.abs(stats.zscore(rolling)) < 4)]

        # Fit models to rolling data
        mle.fit(rolling, sort_state_seq=True, verbose=True)
        jump.fit(rolling, sort_state_seq=True, get_hmm_params=True, verbose=True)

       # Decode last state
        decoded_states['mle'].append(mle.decode(rolling)[-1])
        decoded_states['jump'].append(jump.state_seq[-1])

    decoded_states['mle'] = np.array(decoded_states['mle'])
    decoded_states['jump'] = np.array(decoded_states['jump'])

    decoded_states['mle_trans'] = (np.diff(decoded_states['mle']) != 0).sum()
    decoded_states['jump_trans'] = (np.diff(decoded_states['jump']) != 0).sum()

    return decoded_states

def plot_decoded_states(decoded_states, logret, savefig=None):
    """ Plot decoded states """
    logret = logret.iloc[-len(decoded_states['jump']):]

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 7), sharex=True)

    # ACF of data and models
    x = logret.index
    ax[0].plot(x, logret, color='black', alpha=0.4)
    ax[1].plot(x, decoded_states['mle'], label='mle', color='lightgrey')
    ax[2].plot(x, decoded_states['jump'], label='jump', color='black')

    # Styling
    titles = ['S&P 500', 'MLE', 'Jump']
    ylabels = [r'$\log(r_t)$', r'$s_t$', r'$s_t$']
    trans_list = ['', f'Total transitions: {decoded_states["mle_trans"]}',
            f'Total transitions: {decoded_states["jump_trans"]}']

    for (i, title, ylabel, trans) in zip(range(len(ax)), titles, ylabels, trans_list):
        ax[i].set_title(title)
        ax[i].set_xlim(left=x.min(), right=x.max())
        ax[i].set_ylabel(ylabel)
        ax[i].text(0.83, 1.01, trans, size=15, color='black', transform=ax[i].transAxes)


    #plt.legend()
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
    decoded_states = rolling_state_decoding(logret, mle, jump, window_len=1700,
                                outlier_corrected=False)



    # Save results
    save = False
    #if save == True:
        #plot_simulated_acf(simulations, logret, n_lags=500, savefig='simulated_abs_acf.png')
        #plot_simulated_acf_outliers(simulations, simulations_outliers, logret, logret_outliers,
        #                            n_lags=500, savefig='simulated_abs_acf_outliers.png')
    #else:
        #plot_simulated_acf(simulations, logret, n_lags=500, savefig=None)
        #plot_simulated_acf_outliers(simulations, simulations_outliers, logret, logret_outliers,
        #                            n_lags=500, savefig=None)
