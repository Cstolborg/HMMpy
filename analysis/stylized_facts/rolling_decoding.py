import copy
import json

import pandas as pd;
from matplotlib.collections import LineCollection

pd.set_option('display.max_columns', 10);
pd.set_option('display.width', 320)
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tqdm
from statsmodels.tsa.stattools import acf
from utils.data_prep import load_long_series_logret
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM
import warnings

warnings.filterwarnings("ignore")


def rolling_state_decoding(logret, mle, jump, window_len=1700, median_window=6,
                           outlier_corrected=False):
    n_obs = len(logret)

    vars = {'states': [], 'posteriors': [], 'states_filtered': []}
    decoded_states = {'mle': copy.deepcopy(vars),
                      'jump': copy.deepcopy(vars)}

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
        decoded_states['mle']['states'].append(mle.decode(rolling)[-1])
        decoded_states['jump']['states'].append(jump.state_seq[-1])

        # Apply median filter
        median_mle = np.median(np.append(decoded_states['mle']['states'][-median_window:], mle.decode(rolling)[-1]))
        decoded_states['mle']['states_filtered'].append(median_mle)
        median_jump = np.median(np.append(decoded_states['jump']['states'][-median_window:], jump.decode(rolling)[-1]))
        decoded_states['jump']['states_filtered'].append(median_jump)



        # Get posteriors
        decoded_states['mle']['posteriors'].append(mle.rolling_posteriors(rolling))
        decoded_states['jump']['posteriors'].append(jump.rolling_posteriors(rolling))

    # Transform lists into numpy arrays
    decoded_states['mle']['posteriors'] = np.array(decoded_states['mle']['posteriors'])
    decoded_states['jump']['posteriors'] = np.array(decoded_states['jump']['posteriors'])
    decoded_states['mle']['states'] = np.array(decoded_states['mle']['states'])
    decoded_states['jump']['states'] = np.array(decoded_states['jump']['states'])

    decoded_states['mle']['trans'] = (np.diff(decoded_states['mle']['states']) != 0).sum()
    decoded_states['jump']['trans'] = (np.diff(decoded_states['jump']['states']) != 0).sum()
    decoded_states['mle']['trans_filtered'] = (np.diff(decoded_states['mle']['states_filtered']) != 0).sum()
    decoded_states['jump']['trans_filtered'] = (np.diff(decoded_states['jump']['states_filtered']) != 0).sum()

    return decoded_states

def plot_decoded_states(decoded_states, logret, savefig=None):
    """ Plot decoded states """
    logret = logret.iloc[-len(decoded_states['jump']['states']):]
    x = mdates.date2num(logret.index.to_pydatetime())

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 7), sharex=True)

    # Styling
    titles = ['MLE', 'Jump']
    ylabels = [r'$\log(r_t)$'] * 2
    trans_list = [f'Total transitions: {decoded_states["mle"]["trans"]}',
                  f'Total transitions: {decoded_states["jump"]["trans"]}']
    models = ['mle', 'jump']

    for (i, title, ylabel, trans, model) in zip(range(len(ax)), titles, ylabels, trans_list, models):
        # Ready colors
        c = ['black' if a == 1 else 'lightgrey' for a in decoded_states[model]['states']]
        lines = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(x[:-1], logret[:-1], x[1:], logret[1:])]
        colored_lines = LineCollection(lines, colors=c, linewidths=(2,))

        ax[i].add_collection(colored_lines)
        ax[i].autoscale_view()

        ax[i].set_title(title)
        ax[i].set_xlim(left=x.min(), right=x.max())
        ax[i].set_ylabel(ylabel)
        ax[i].xaxis_date()
        ax[i].text(0.83, 1.01, trans, size=15, color='black', transform=ax[i].transAxes)

    #plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()


def plot_decoded_states_filter(decoded_states, logret, savefig=None):
    """ Plot decoded states """
    logret = logret.iloc[-len(decoded_states['jump']['states']):]
    x = mdates.date2num(logret.index.to_pydatetime())

    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 7), sharex=True)

    # Styling
    titles = ['MLE', 'Jump']
    ylabels = [r'$\log(r_t)$'] * 2
    trans_list = [f'Total transitions: {decoded_states["mle"]["trans_filtered"]}',
                  f'Total transitions: {decoded_states["jump"]["trans_filtered"]}']
    models = ['mle', 'jump']

    for (i, title, ylabel, trans, model) in zip(range(len(ax)), titles, ylabels, trans_list, models):
        # Ready colors
        c = ['black' if a == 1 else 'lightgrey' for a in decoded_states[model]['states_filtered']]
        lines = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(x[:-1], logret[:-1], x[1:], logret[1:])]
        colored_lines = LineCollection(lines, colors=c, linewidths=(2,))

        ax[i].add_collection(colored_lines)
        ax[i].autoscale_view()

        ax[i].set_title(title)
        ax[i].set_xlim(left=x.min(), right=x.max())
        ax[i].set_ylabel(ylabel)
        ax[i].xaxis_date()
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

    plot_decoded_states(decoded_states, logret, savefig=None)

    # Save results
    save = False
    if save == True:
        #plot_decoded_states(decoded_states, logret, savefig='decoded_states_filter.png')
        plot_decoded_states(decoded_states, logret, savefig='decoded_states.png')
        plot_decoded_states_filter(decoded_states, logret, savefig='decoded_states_filter.png')

        with open('./output_data/rolling_decoded_states.json', 'w') as f:
            json.dump(decoded_states, f, indent=4)
    else:
        plot_decoded_states(decoded_states, logret, savefig=None)
        plot_decoded_states_filter(decoded_states, logret, savefig=None)