import copy

import numpy as np
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from utils.hmm_sampler import SampleHMM
from analysis.model_convergence.simulation import test_model_convergence, plot_simulated_model_convergence
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM


def fit_t_dist():
    X = stats.t.rvs(loc=sampler.mu[0], scale=sampler.std[0], size=int(2e6), df=5).reshape(2000, -1)

    df = pd.DataFrame()
    sample_lengths = (250, 500, 1000, 2000)
    for sample_length in sample_lengths:
        data = {'Fitted t': {'mean': [], 'std': []},
                'Fitted normal': {'mean': [], 'std': []}}
        for seq in tqdm.tqdm(X.T):
            seq = seq[:sample_length]
            t_stats = stats.t.fit(seq)  # Returns df, loc, scale
            norm_stats = stats.norm.fit(seq)  # Returns loc, scale
            data['Fitted t']['mean'].append(t_stats[1])
            data['Fitted t']['std'].append(t_stats[2])
            data['Fitted normal']['mean'].append(norm_stats[0])
            data['Fitted normal']['std'].append(norm_stats[1])

        for model in data.keys():
            df_temp = pd.DataFrame(data[model])
            df_temp['model'] = model
            df_temp['Simulation length'] = sample_length
            df = df.append(df_temp)

    # Add true values to data
    for sample_length in sample_lengths:
        true_data = \
            {'model': 'true',
             'mean': sampler.mu[0],
             '$std$': sampler.std[0],
             'Simulation length': sample_length
             }
        df = df.append(pd.DataFrame(true_data, index=[0]))

    return df

def plot_simulated_dist_convergence_box(df, sampler, savefig=None):
    # Plotting
    plt.rcParams.update({'font.size': 25})
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # Set symbols on y-axis
    symbol_list = ['$\mu_1$', '$\sigma_1$']
    cols = ['mean', 'std']
    colors = ['black', 'lightgrey']  # ['#0051a2', '#97964a', '#ffd44f', '#f4777f', '#93003a']

    to_plot = df[df['model'] != 'true']
    k = 0  # columns indexer
    for j in range(2):
        sns.boxplot(data=to_plot, x='Simulation length', y=cols[j],
                                hue='model', showfliers=False, ax=axes[j])
        axes[j].set_ylabel(symbol_list[j])

    # Plot true values
    for i, true_val in enumerate([sampler.mu[0], sampler.std[0]]):
        axes[i].axhline(y=true_val, ls="--", color="black", label='True')

    # Remove seaborn legends and x-labels
    for ax in axes.flatten():
        ax.legend([], [], frameon=False)
        ax.set_xlabel("")

    axes[0].legend(loc='lower right', fontsize=15)

    # Set ylims
    #axes[-1, 0].set_ylim(0.75, 1.01)
    #axes[-1, 1].set_ylim(top=1.01)

    axes[-1].set_xlabel('Simulation length')
    axes[-1].set_xlabel('Simulation length')
    plt.tight_layout()

    if not savefig == None:
        plt.savefig('./images/' + savefig)
    plt.show()

if __name__ == '__main__':
    """
    1. Simulate t-distribution from HMM
    2. Fit both conditional t-distributions to normal distribution assuming states are known.
    3. Plot precision
    """
    path = './output_data/'
    sampler = SampleHMM()

    df = fit_t_dist()
    plot_simulated_dist_convergence_box(df, sampler, savefig='theoretical_fit_t_dist.png')




