import multiprocessing
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tqdm
from sklearn.metrics import confusion_matrix

from models.hidden_markov.hmm_jump import JumpHMM
from utils.hmm_sampler import SampleHMM


def plot_jump_penalties(bac_outer, sample_lengths, penalties, savefig=None):
    # Plot line
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xscale('log', basex=2)
    for i, sample_length in enumerate(sample_lengths):
        ax.plot(penalties, np.mean(bac_outer[i], axis=1), label=str(sample_length))

    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('Balanced Accuracy')

    ax.legend()
    plt.tight_layout()
    if not savefig == None:
        plt.savefig('./images/' + savefig) # Remember to change name when running with different penalties.
    plt.show()

def plot_jump_penalties_box(bac_outer, sample_lengths, penalties, save=True):
    # Plot boxplot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(bac_outer[-1], showfliers=False)

    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('Balanced Accuracy')

    plt.xticks(list(range(len(penalties))), penalties)
    plt.tight_layout()
    plt.savefig('./images/jump_penalties_box_feature_set_3_all.png')  # Remember to change name when running with different penalities.
    plt.show()

def plot_optimal_penalties_box(df, savefig=None):
    # Plotting
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    sample_lengths = [250, 500, 1000, 2000]
    for i in range(4):
        to_plot = df[df['sample_length'] == sample_lengths[i]]
        sns.boxplot(x='$\lambda$', y='BAC', data=to_plot,
                    showfliers=False, ax=axes[i], hue='sample_length')
        axes[i].set_ylabel('BAC')

    # Remove seaborn legends and x-labels
    for ax in axes.flatten():
        ax.legend([], [], frameon=False)
        ax.legend(loc='lower left', fontsize=15)
        ax.set_xlabel("")
        ax.set_ylim(bottom=0.45, top=1.01)

    axes[-1].set_xlabel('$\lambda$')

    # Set ylims
    #axes[2, 0].set_ylim(0.75, 1.01)
    #axes[2, 1].set_ylim(0.75, top=1.01)

    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

def compute_penalties(X, true_states, viterbi_states, penalties, sample_lengths=[250, 500, 1000, 2000]):

    # Create pool object for multiprocessing
    # pool = Pool(processes=multiprocessing.cpu_count()-6)  # Spare 2 cpu's

    bac_outer = []

    for sample_length in sample_lengths:
        print(f'Computing BAC for sample length={sample_length}')
        n_samples = sample_length

        # Slice data
        X_current, true_states_current = X[:n_samples], true_states[:n_samples]
        model = JumpHMM(n_states=2, window_len=(6, 14), random_state=42)

        # Setup partial func and create map object to iterate over different penalties
        mapfunc = partial(model.bac_score_nd, X_current, true_states_current)
        # bac_temp = pool.map(mapfunc, penalties)  # list containing n_sequences bac scores for each penalty
        bac_temp = list(map(mapfunc, penalties))  # list containing n_sequences bac scores for each penalty
        bac_outer.append(bac_temp)

    # pool.close()

    # Compute viterbi accuracy with true params in the full sample
    # TODO this is not used for anything currently??
    for seq in range(X.shape[1]):
        conf_matrix = confusion_matrix(true_states[:1000, seq], viterbi_states[:1000, seq])
        keep_idx = conf_matrix.sum(axis=1) != 0
        conf_matrix = conf_matrix[keep_idx]

        tp = np.diag(conf_matrix)
        fn = conf_matrix.sum(axis=1) - tp
        tpr = tp / (tp + fn)
        viterbi_bac = np.mean(tpr)

    return bac_outer


if __name__ == '__main__':
    sampler = SampleHMM(n_states=2, random_state=42)

    #n_sequences = 1000
    #n_samples = 2000
    #X, viterbi_states, true_states = sampler.sample_with_viterbi(n_samples, n_sequences)

    # Load sampled data, sampled states and viterbi states
    path = '../../analysis/model_convergence/output_data/'
    X = np.load(path + 'sampled_returns.npy')
    viterbi_states = np.load(path + 'sampled_viterbi_states.npy')
    true_states = np.load(path + 'sampled_true_states.npy')

    # Define the grid to search over
    penalties = np.logspace(-2, 7, num=10, base=2)
    sample_lengths = [250, 500, 1000, 2000]

    # Compute Bac across sample lens and for all penalties
    bac_outer = compute_penalties(X, true_states, viterbi_states, penalties, sample_lengths=sample_lengths)

    #Turn bac_outer into a pd.DataFrame
    df = pd.DataFrame()
    for i in range(len(sample_lengths)):
        df_temp = pd.DataFrame(np.array(bac_outer[0]).T, columns=penalties)
        df_temp['sample_length'] = sample_lengths[i]
        df = df.append(df_temp)

    df = df.melt(id_vars=['sample_length'], var_name='$\lambda$', value_name='BAC')

    # Plotting
    plot_jump_penalties(bac_outer, sample_lengths, penalties, savefig=None)

    plot_optimal_penalties_box(df, savefig='jump_penalties_feature_set_3_box.png')













    """
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    plt.plot(penalties, bac)
    plt.show()
    """


    """
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(X[:, 2], label='Sample returns')
    ax[1].plot(viterbi_states[:, 2], label='Sampled viterbi states', ls='--')
    ax[1].plot(true_states[:, 2], label='Sampled true states', ls='solid')
    ax[1].plot(model1_seq, label='Jump predictions', ls='dotted', lw=3)
    plt.legend()
    plt.show()
    """


    '''
    for sample_length in tqdm.tqdm(sample_lengths):
        n_samples = sample_length
        n_sequences = 1000
        X, viterbi_states, true_states = sampler.sample_with_viterbi(n_samples, n_sequences)
        bac_inner = []

        for penalty in penalties:
            model = JumpHMM(n_states=2, jump_penalty=penalty, random_state=42)
            bac_temp = model.bac_score_nd(X, true_states, jump_penalty=penalty)
            bac_inner.append(bac_temp)
        bac_outer.append(bac_inner)
    '''