import copy

import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from utils.data_prep import load_long_series_logret
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM


def plot_something():
    fig, ax = plt.subplots()

    # Rest of plot
    plt.show()
    pass


if __name__ == '__main__':
    path = '../../analysis/stylized_facts/output_data/'
    df = pd.read_csv(path + 'rolling_estimations.csv')


    data_table = df.groupby(['window_len', 'model']).mean().sort_index(ascending=[True, False])
    print(data_table)


    # Do rest of analysis here

    plot_something()

