import copy

import numpy as np
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import tqdm
import matplotlib.pyplot as plt

from utils.hmm_sampler import SampleHMM
from analysis.model_convergence.simulation import test_model_convergence, plot_simulated_model_convergence, plot_simulated_model_convergence_box
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    jump = JumpHMM(n_states=2, jump_penalty=16, window_len=(6, 14))
    mle = EMHiddenMarkov(n_states=2)
    sampler = SampleHMM(n_states=2)

    path = '../../analysis/model_convergence/output_data/'
    X = np.load(path + 'sampled_t_returns.npy')
    true_states = np.load(path + 'sampled_t_true_states.npy')

    #df = pd.read_csv(path + 'simulation_t.csv')
    df = test_model_convergence(jump, mle, sampler, X, true_states, sample_lengths=(250, 500, 1000, 2000))

    # Summarize results
    data_table = df.groupby(['Simulation length', 'model']).mean().sort_index(ascending=[True, False])
    print(data_table)
    print('N times not fitted \n', df[df['is_fitted'] == False].groupby('model')['is_fitted'].count() )

    # Show results after removing sequences with only 1 state
    df2 = df[df['two_states'] == True]


    save = False
    if save == True:
        plot_simulated_model_convergence(df, sampler, savefig='simulation_t.png')
        plot_simulated_model_convergence_box(df, sampler, savefig='simulation_t_box.png')
        df.to_csv(path + 'simulation_t.csv', index=False)
        data_table.round(4).to_latex(path + 'simulation_t.tex', escape=False)

        plot_simulated_model_convergence(df2, sampler, savefig='simulation_t_2states.png')
        plot_simulated_model_convergence_box(df2, sampler, savefig='simulation_t_box_2states.png')
    else:
        plot_simulated_model_convergence(df, sampler, savefig=None)
        plot_simulated_model_convergence_box(df, sampler, savefig=None)

        plot_simulated_model_convergence(df2, sampler, savefig=None)
        plot_simulated_model_convergence_box(df2, sampler, savefig=None)
