import copy

import numpy as np
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import tqdm
import matplotlib.pyplot as plt

from scipy import stats

from utils.hmm_sampler import SampleHMM
from analysis.model_convergence.simulation import test_model_convergence, plot_simulated_model_convergence
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM

if __name__ == '__main__':
    """
    1. Simulate t-distribution from HMM
    2. Fit both conditional t-distributions to normal distribution assuming states are known.
    3. Plot precision
    """

    #X = np.load(path + 'sampled_t_returns.npy')