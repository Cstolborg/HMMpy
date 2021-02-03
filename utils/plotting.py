import numpy as np
import matplotlib.pyplot as plt

def plot_samples_states(samples, states):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(samples, label='Sample returns', )
    ax[1].plot(states, label='Sampled true states', ls='dotted')
    plt.legend()
    plt.show()

def plot_samples_states_viterbi(samples, viterbi_states, true_states):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(samples, label='Sample returns')
    ax[1].plot(viterbi_states, label='Sampled viterbi states', ls='dotted')
    ax[1].plot(true_states, label='Sampled true states', ls='dotted')
    plt.legend()
    plt.show()

def plot_posteriors_states(posteriors, states, true_regimes):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
    ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
    ax[1].plot(states, label='Predicted states', ls='dotted')
    ax[1].plot(true_regimes, label='True states', ls='dashed')

    plt.legend()
    plt.show()