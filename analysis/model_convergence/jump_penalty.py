import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tqdm
from multiprocessing import Pool
from functools import partial

from models.hidden_markov.hmm_jump import JumpHMM
from utils.hmm_sampler import SampleHMM


if __name__ == '__main__':
    model = JumpHMM(n_states=2, jump_penalty=0, random_state=42)
    sampler = SampleHMM(n_states=2, random_state=42)

    pool = Pool()
    # result = pool.map(mapfunc, [X]*20)


    #penalties = [0.01 ,0.1, 1, 10, 25, 50, 100]
    penalties = np.logspace(-2, 7, num=20, base=2)
    bac_outer = []
    sample_lengths = [250, 500, 1000, 2000]
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

    #model = JumpHMM(n_states=2, jump_penalty=100000, random_state=42)
    #mapfunc = partial(model.bac_score_nd, X=X, y_true=true_states)
    #bac_temp = pool.map(mapfunc, penalties)

    #def compute_penalty(penalty, X=X, true_states=true_states):
    #    return model.bac_score_nd(X, true_states, penalty)
    #result = pool.map(compute_penalty, penalties)



    fig, ax = plt.subplots(2,1)
    ax[0].set_xscale('log')
    for i, sample_length in enumerate(sample_lengths):
        ax[0].plot(penalties, np.mean(bac_outer[i], axis=1), label=str(sample_length))

    ax[1].boxplot(bac_outer[-1], showfliers=False)
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