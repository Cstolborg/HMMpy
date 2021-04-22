import warnings
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import matplotlib.pyplot as plt

from utils.hmm_sampler import SampleHMM
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

def test_model_convergence(jump, mle, sampler, X, Y_true, sample_lengths=(250, 500, 1000, 2000)):
    """ Test model convergence on simulated data """
    df = pd.DataFrame()

    # Set attributes to be saved in simulation
    cols = {
        '$\mu_1$': [], '$\mu_2$': [],
        '$\sigma_1$': [], '$\sigma_2$': [],
        '$q_{11}$': [], '$q_{22}$': [],
        'BAC': [], 'is_fitted': [],
        'two_states': []
    }

    # Compute models params for each sample length
    for sample_length in sample_lengths:
        print(f'Computing values for sample length = {sample_length}')
        # Load empty dictionaries for each run
        data = {'jump': copy.deepcopy(cols),
                'mle': copy.deepcopy(cols)
                }

        # Iterate through each sequence
        for seq in tqdm.tqdm(range(X.shape[1])):
            # Slice generated data
            x = X[:sample_length, seq]
            y_true = Y_true[:sample_length, seq]

            # Check if both states are present
            if len(np.unique(y_true)) < 2:
                data['mle']['two_states'].append(False)
                data['jump']['two_states'].append(False)
            else:
                data['mle']['two_states'].append(True)
                data['jump']['two_states'].append(True)

            jump.fit(x, sort_state_seq=True, get_hmm_params=True, verbose=True)
            mle.fit(x, sort_state_seq=True, verbose=True)

            data['jump']['$\mu_1$'].append(jump.mu[0])
            data['jump']['$\mu_2$'].append(jump.mu[1])
            data['jump']['$\sigma_1$'].append(jump.std[0])
            data['jump']['$\sigma_2$'].append(jump.std[1])
            data['jump']['$q_{11}$'].append(jump.tpm[0, 0])
            if len(jump.tpm) > 1:
                data['jump']['$q_{22}$'].append(jump.tpm[1, 1])
            else:
                data['jump']['$q_{22}$'].append(0)

            data['jump']['BAC'].append(jump.bac_score(x, y_true))
            data['jump']['is_fitted'].append(jump.is_fitted)

            data['mle']['$\mu_1$'].append(mle.mu[0])
            data['mle']['$\mu_2$'].append(mle.mu[1])
            data['mle']['$\sigma_1$'].append(mle.std[0])
            data['mle']['$\sigma_2$'].append(mle.std[1])
            data['mle']['$q_{11}$'].append(mle.tpm[0, 0])
            if len(mle.tpm) > 1:
                data['mle']['$q_{22}$'].append(mle.tpm[1, 1])
            else:
                data['mle']['$q_{22}$'].append(0)

            data['mle']['BAC'].append(mle.bac_score(x, y_true))
            data['mle']['is_fitted'].append(mle.is_fitted)

        for model in data.keys():
            df_temp = pd.DataFrame(data[model])
            df_temp['model'] = model
            df_temp['Simulation length'] = sample_length
            df = df.append(df_temp)

    # Add true values to data
    for sample_length in sample_lengths:
        true_data = \
            {'model': 'true',
             '$\mu_1$': sampler.mu[0],
             '$\mu_2$': sampler.mu[1],
             '$\sigma_1$': sampler.std[0],
             '$\sigma_2$': sampler.std[1],
             '$q_{11}$': sampler.tpm[0, 0],
             '$q_{22}$': sampler.tpm[1, 1],
             'Simulation length': sample_length
             }
        df = df.append(pd.DataFrame(true_data, index=[0]))

    return df

def plot_simulated_model_convergence(df, sampler, savefig=None):
    # Plotting
    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

    # Set symbols on y-axis
    symbol_list = [['$\mu_1$', '$\mu_2$'],
                   ['$\sigma_1$', '$\sigma_2$'],
                   ["$q_{11}$", "$q_{22}$"]]

    models = ['jump', 'mle']

    for model in models:
        # Slice df to get desired column
        to_plot = df[df['model'] == model]
        to_plot = to_plot.groupby('Simulation length').mean()
        k = 0  # columns indexer
        for i in range(3):
            for j in range(2):
                to_plot1 = to_plot.iloc[:, k]  # column to plot
                ax[i, j].plot(to_plot1, label=str(model))
                ax[i, j].set_ylabel(symbol_list[i][j])
                k += 1

    # Plot true values
    for i in range(2):
        ax[0, i].axhline(y=sampler.mu[i], ls="--", color="black", label='True')
        ax[1, i].axhline(y=sampler.std[i], ls="--", color="black", label='True')
        ax[2, i].axhline(y=sampler.tpm[i, i], ls="--", color="black", label='True')

    ax[0, 0].legend(fontsize=15)

    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

def plot_simulated_model_convergence_box(df, sampler, savefig=None):
    # Plotting
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

    # Set symbols on y-axis
    symbol_list = [['$\mu_1$', '$\mu_2$'],
                   ['$\sigma_1$', '$\sigma_2$'],
                   ["$q_{11}$", "$q_{22}$"]]

    to_plot = df[df['model'] != 'true']
    k = 0  # columns indexer
    for i in range(3):
        for j in range(2):
            sns.boxplot(data=to_plot, x='Simulation length', y=symbol_list[i][j],
                                    hue='model', showfliers=False, ax=axes[i, j])
            axes[i, j].set_ylabel(symbol_list[i][j])
            k += 1

    # Plot true values
    for i in range(2):
        axes[0, i].axhline(y=sampler.mu[i], ls="--", color="black", label='True')
        axes[1, i].axhline(y=sampler.std[i], ls="--", color="black", label='True')
        axes[2, i].axhline(y=sampler.tpm[i, i], ls="--", color="black", label='True')

    # Remove seaborn legends and x-labels
    for ax in axes.flatten():
        ax.legend([], [], frameon=False)
        ax.set_xlabel("")

    axes[-1, 0].legend(loc='lower right', fontsize=15)

    # Set ylims
    axes[-1, 0].set_ylim(0.75, 1.01)
    axes[-1, 1].set_ylim(0.75, top=1.01)

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
    #df = test_model_convergence(jump, mle, sampler, X, true_states, sample_lengths=(250, 500, 1000, 2000))

    # Summarize results
    data_table = df.groupby(['Simulation length', 'model']).mean().sort_index(ascending=[True, False])
    print(data_table)
    print('N times not fitted \n', df[df['is_fitted'] == False].groupby('model')['is_fitted'].count() )

    # Show results after removing sequences with only 1 state
    df2 = df[df['two_states'] == True]

    save = False
    if save == True:
        plot_simulated_model_convergence(df, sampler, savefig='simulation_normal.png')
        plot_simulated_model_convergence_box(df, sampler, savefig='simulation_normal_box.png')
        df.to_csv(path + 'simulation_normal.csv', index=False)
        data_table.round(4).to_latex(path + 'simulation_normal.tex', escape=False)

        plot_simulated_model_convergence(df2, sampler, savefig='simulation_normal_2states.png')
        plot_simulated_model_convergence_box(df2, sampler, savefig='simulation_normal_box_2states.png')
    else:
        plot_simulated_model_convergence(df, sampler, savefig=None)
        plot_simulated_model_convergence_box(df, sampler, savefig=None)

        plot_simulated_model_convergence(df2, sampler, savefig=None)
        plot_simulated_model_convergence_box(df2, sampler, savefig=None)

