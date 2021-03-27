import copy

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from utils.hmm_sampler import SampleHMM
from analysis.model_convergence.simulation import test_model_convergence, plot_simulated_model_convergence
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

    df = pd.read_csv(path + 'simulation_t.csv')
    df = test_model_convergence(jump, mle, sampler, X, sample_lengths=(250, 500, 1000, 2000))

    plot_simulated_model_convergence(df, sampler, savefig='simulation_t.png')


    data_table = df.groupby(['sample_size', 'model']).mean().sort_index(ascending=[True, False])
    data_table.to_latex(path + 'simulation_t.tex')
    print(data_table)