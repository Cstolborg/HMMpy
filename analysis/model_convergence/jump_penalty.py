import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tqdm

from models.hidden_markov.hmm_jump import JumpHMM
from utils.hmm_sampler import SampleHMM


if __name__ == '__main__':
    model = JumpHMM(n_states=2, jump_penalty=0, random_state=42)
    sampler = SampleHMM(n_states=2, random_state=42)

    n_samples = 1000
    n_sequences = 1000
    X, viterbi_states, true_states = sampler.sample_with_viterbi(n_samples, n_sequences)

    bac = []
    penalties = [0.01 ,0.1, 1, 10, 25, 50, 100]
    for penalty in tqdm.tqdm(penalties):
        model = JumpHMM(n_states=2, jump_penalty=penalty, random_state=42)
        bac_temp = model.bac_score_nd(X, viterbi_states, jump_penalty=penalty)
        bac.append(bac_temp)

    fig, ax = plt.subplots(2,1)
    ax[0].set_xscale('log')
    ax[0].plot(penalties, np.mean(bac, axis=1))
    ax[1].boxplot(bac)
    ax[1].set_xticklabels(penalties)
    plt.suptitle(f"n_samples = {n_samples}")
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