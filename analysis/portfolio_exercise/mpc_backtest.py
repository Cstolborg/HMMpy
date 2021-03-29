import numpy as np; np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.finance.backtest import Backtester
from utils.data_prep import load_data_get_ret , load_data_get_logret, load_data
from utils.plotting.plot_asset_vals import plot_performance

import warnings
warnings.filterwarnings('ignore')





if __name__ == "__main__":
    path = '../../analysis/portfolio_exercise/output_data/'
    df_logret = load_data_get_logret()
    X = df_logret["MSCI World"]

    model1 = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)
    backtester = Backtester()

    #preds, cov = backtester.rolling_preds_cov_from_hmm(X, df_logret, model1, window_len=1700, shrinkage_factor=(0.3, 0.3), verbose=True)
    #np.save(path + 'rolling_preds.npy', preds)
    #np.save(path + 'rolling_cov.npy', cov)

    df_ret = load_data_get_ret()
    preds = np.load(path + 'rolling_preds.npy')
    cov = np.load(path + 'rolling_cov.npy')

    #weights, port_val, gamma = backtester.backtest_mpc(df_ret, preds, cov, short_cons='LLO')

    #np.save(path + 'mpc_weights.npy', weights)
    #np.save(path + 'port_val.npy', port_val)
    #np.save(path + 'gamma.npy', gamma)

    port_val = np.load(path + 'port_val.npy')
    weights = np.load(path + 'mpc_weights.npy')
    df = load_data()

    metrics = backtester.performance_metrics(df, port_val, compare_assets=True)

    print(metrics)
    #print('transaction costs:', (1-backtester.trans_cost).prod())
    #print('highest trans cost', backtester.trans_cost.max())

    df = df.iloc[-len(port_val):]


    save = False
    if save == True:
        metrics.round(4).to_latex(path + 'asset_performance.tex')
        plot_performance(df, port_val, weights, save=True)
    else:
        plot_performance(df, port_val, weights, save=False)