import copy

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from utils.hmm_sampler import SampleHMM
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM

import warnings
warnings.filterwarnings("ignore")


def test_model_convergence(jump, mle, sampler, X, sample_lengths=(250, 500, 1000, 2000)):
    """ Test model convergence on simulated data """
    df = pd.DataFrame()

    # Set attributes to be saved in simulation
    cols = {
        '$\mu_1$': [], '$\mu_2$': [],
        '$\std_1$': [], '$\std_2$': [],
        '$q_11$': [], '$q_22$': []
    }

    # Compute models params for each sample length
    for sample_length in tqdm.tqdm(sample_lengths):
        # Load empty dictionaries for each run
        data = {'jump': copy.deepcopy(cols),
                'mle': copy.deepcopy(cols)
                }

        # Iterate through each sequence
        for seq in tqdm.tqdm(range(X.shape[1])):
            jump.fit(X[:sample_length, seq], sort_state_seq=True, get_hmm_params=True, verbose=True)
            mle.fit(X[:sample_length, seq], sort_state_seq=True, verbose=True)

            data['jump']['$\mu_1$'].append(jump.mu[0])
            data['jump']['$\mu_2$'].append(jump.mu[1])
            data['jump']['$\std_1$'].append(jump.std[0])
            data['jump']['$\std_2$'].append(jump.std[1])
            data['jump']['$q_11$'].append(jump.tpm[0, 0])
            if len(jump.tpm) > 1:
                data['jump']['$q_22$'].append(jump.tpm[1, 1])
            else:
                data['jump']['$q_22$'].append(0)

            data['mle']['$\mu_1$'].append(mle.mu[0])
            data['mle']['$\mu_2$'].append(mle.mu[1])
            data['mle']['$\std_1$'].append(mle.std[0])
            data['mle']['$\std_2$'].append(mle.std[1])
            data['mle']['$q_11$'].append(mle.tpm[0, 0])
            if len(mle.tpm) > 1:
                data['mle']['$q_22$'].append(mle.tpm[1, 1])
            else:
                data['mle']['$q_22$'].append(0)

        for model in data.keys():
            df_temp = pd.DataFrame(data[model])
            df_temp['model'] = model
            df_temp['sample_size'] = sample_length
            df = df.append(df_temp)

    # Add true values to data
    for sample_length in sample_lengths:
        true_data = \
            {'model': 'true',
             '$\mu_1$': sampler.mu[0],
             '$\mu_2$': sampler.mu[1],
             '$\std_1$': sampler.std[0],
             '$\std_2$': sampler.std[1],
             '$q_11$': sampler.tpm[0, 0],
             '$q_22$': sampler.tpm[1, 1],
             'sample_size': sample_length
             }
        df = df.append(pd.DataFrame(true_data, index=[0]))

    return df

def plot_simulated_model_convergence(df, sampler, savefig=None):
    # Plotting
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

    # Set symbols on y-axis
    symbol_list = [['$\mu_1$', '$\mu_2$'],
                   ['$\sigma_1$', '$\sigma_2$'],
                   ["$p_{11}$", "$p_{22}$"]]

    models = ['jump', 'mle']
    colors = ['black', 'lightgrey']  # ['#0051a2', '#97964a', '#ffd44f', '#f4777f', '#93003a']

    for (model, color) in zip(models, colors):
        # Slice df to get desired column
        to_plot = df[df['model'] == model]
        to_plot = to_plot.groupby('sample_size').mean()
        k = 0  # columns indexer
        for i in range(3):
            for j in range(2):
                to_plot1 = to_plot.iloc[:, k]  # column to plot
                ax[i, j].plot(to_plot1, label=str(model), color=color)
                ax[i, j].set_ylabel(symbol_list[i][j])
                k += 1

    # Plot true values
    for i in range(2):
        ax[0, i].axhline(y=sampler.mu[i], ls="--", color="black", label='True')
        ax[1, i].axhline(y=sampler.std[i], ls="--", color="black", label='True')
        ax[2, i].axhline(y=sampler.tpm[i, i], ls="--", color="black", label='True')

    ax[0, 0].legend()

    ax[-1, 0].set_xlabel('Simulation length')
    ax[-1, 1].set_xlabel('Simulation length')
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()


if __name__ == '__main__':
    jump = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14))
    mle = EMHiddenMarkov(n_states=2)
    sampler = SampleHMM(n_states=2)

    path = '../../analysis/model_convergence/output_data/'
    X = np.load(path + 'sampled_returns.npy')
    true_states = np.load(path + 'sampled_true_states.npy')

    df = pd.read_csv(path + 'simulation_normal.csv')
    #df = test_model_convergence(jump, mle, sampler, X, sample_lengths=(250, 500, 1000, 2000))

    plot_simulated_model_convergence(df, sampler, savefig='simulation_normal.png')


    data_table = df.groupby(['sample_size', 'model']).mean().sort_index(ascending=[True, False])
    data_table.to_latex(path + 'simulation_normal.tex')
    print(data_table)