import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def plot_samples_states(samples, states, show=True):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(samples, label='Sample returns', )
    ax[1].plot(states, label='Sampled true states', ls='dotted')
    plt.legend()

    if show is True:
        plt.show()
    return fig, ax

def plot_samples_states_viterbi(samples, viterbi_states, true_states, show=True):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(samples, label='Sample returns')
    ax[1].plot(viterbi_states, label='Sampled viterbi states', ls='dotted')
    ax[1].plot(true_states, label='Sampled true states', ls='solid')
    plt.legend()

    if show == True:
        plt.show()
    return fig, ax

def plot_posteriors_states(posteriors, states, true_regimes, show=True):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
    ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
    ax[1].plot(states, label='Predicted states', ls='dotted')
    ax[1].plot(true_regimes, label='True states', ls='dashed')

    plt.legend()
    if show == True:
        plt.show()
    return fig, ax


def plot_2state_hmm_params(tpm, mu, std, true_tpm, true_mu, true_std):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 7), sharex=True)
    ax[0, 0].plot(tpm[:, 0], label='P(1)', )
    ax[0, 0].plot([true_tpm[0, 0]] * len(tpm), label='True P1', ls='dashed')
    ax[0, 1].plot(tpm[:, 1], label='P(2)', )
    ax[0, 1].plot([true_tpm[1, 1]] * len(tpm), label='True P2', ls='dashed')
    # ax[0].legend()

    ax[1, 0].plot(mu[:, 0])
    ax[1, 0].plot([true_mu[0]] * len(mu), ls='dashed')
    ax[1, 1].plot(mu[:, 1])
    ax[1, 1].plot([true_mu[1]] * len(mu), ls='dashed')

    ax[2, 0].plot(std[:, 0])  # label='True states', ls='dashed')
    ax[2, 0].plot([true_std[0]] * len(std), ls='dashed')
    ax[2, 1].plot(std[:, 1])  # label='True states', ls='dashed')
    ax[2, 1].plot([true_std[1]] * len(std), ls='dashed')

    plt.show()