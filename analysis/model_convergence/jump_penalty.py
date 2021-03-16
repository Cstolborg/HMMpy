import multiprocessing
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.metrics import confusion_matrix

from models.hidden_markov.hmm_jump import JumpHMM
from utils.hmm_sampler import SampleHMM



if __name__ == '__main__':
    sampler = SampleHMM(n_states=2, random_state=42)

    #n_sequences = 1000
    #n_samples = 2000
    #X, viterbi_states, true_states = sampler.sample_with_viterbi(n_samples, n_sequences)

    path = '../../analysis/model_convergence/output_data/'
    X = np.load(path + 'sampled_returns.npy')
    viterbi_states = np.load(path + 'sampled_viterbi_states.npy')
    true_states = np.load(path + 'sampled_true_states.npy')

    # Create pool object for multiprocessing
    pool = Pool(processes=multiprocessing.cpu_count()-6)  # Spare 2 cpu's
    penalties = np.logspace(-2, 7, num=10, base=2)
    bac_outer = []
    sample_lengths = [250, 500, 1000, 2000]

    for sample_length in tqdm.tqdm(sample_lengths):
        n_samples = sample_length

        # Slice data
        X_current, true_states_current = X[:n_samples], true_states[:n_samples]
        model = JumpHMM(n_states=2, jump_penalty=1000, window_len=(6, 14), random_state=42)

        # Setup partial func and create map object to iterate over different penalties
        mapfunc = partial(model.bac_score_nd, X_current, true_states_current)
        bac_temp = pool.map(mapfunc, penalties)  # list containing n_sequences bac scores for each penalty
        bac_outer.append(bac_temp)

    pool.close()

    # Compute viterbi accuracy with true params in the full sample
    for seq in range(X.shape[1]):
        conf_matrix = confusion_matrix(true_states[:1000, seq], viterbi_states[:1000, seq])
        keep_idx = conf_matrix.sum(axis=1) != 0
        conf_matrix = conf_matrix[keep_idx]

        tp = np.diag(conf_matrix)
        fn = conf_matrix.sum(axis=1) - tp
        tpr = tp / (tp + fn)
        viterbi_bac = np.mean(tpr)

    # Plot line
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xscale('log', basex=2)
    for i, sample_length in enumerate(sample_lengths):
        ax.plot(penalties, np.mean(bac_outer[i], axis=1), label=str(sample_length))

    ax.legend()

    plt.suptitle(f"Selecting the jump penalty")
    plt.tight_layout()
    plt.show()

    #Plot boxplot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(bac_outer[-1], showfliers=False)

    plt.xticks(list(range(len(penalties))), penalties)
    plt.suptitle(f"Selecting the jump penalty")
    plt.tight_layout()
    plt.show()

















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