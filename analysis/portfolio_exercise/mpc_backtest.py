import numpy as np; np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.max_columns', 10); pd.set_option('display.width', 320)

from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.finance.backtest import Backtester
from utils.data_prep import load_data_get_ret , load_data_get_logret, load_data
from analysis.portfolio_exercise.data_description import plot_performance

import warnings
warnings.filterwarnings('ignore')





if __name__ == "__main__":
    path = '../../analysis/portfolio_exercise/output_data/'

    # Get log-returns - used in times series model
    df_logret = load_data_get_logret()
    X = df_logret["S&P 500 "]

    # Instantiate models to test and backtester
    model1 = EMHiddenMarkov(n_states=2, init="random", random_state=42, epochs=20, max_iter=100)
    backtester = Backtester()

    # Uncomment this section to perform new backtest - generating forecast distributions
    # Leave commented to used existing preds and covariances from file
    #preds, cov = backtester.rolling_preds_cov_from_hmm(X, df_logret, model1, window_len=1700, shrinkage_factor=(0.3, 0.3), verbose=True)
    #np.save(path + 'rolling_preds.npy', preds)
    #np.save(path + 'rolling_cov.npy', cov)

    # Leave uncomennted to use forecast distributions from file
    preds = np.load(path + 'rolling_preds.npy')
    cov = np.load(path + 'rolling_cov.npy')

    # Get actual returns - used to test performance of trading strategy
    df_ret = load_data_get_ret()

    # Use forecast distribution to test trading strategy
    #weights, port_val, gamma = backtester.backtest_mpc(df_ret, preds, cov, short_cons='LLO')
    #np.save(path + 'mpc_weights.npy', weights)
    #np.save(path + 'port_val.npy', port_val)
    #np.save(path + 'gamma.npy', gamma)

    # Leave uncommented to use previously tested trading strategy from file
    port_val = np.load(path + 'port_val.npy')
    weights = np.load(path + 'mpc_weights.npy')
    df = load_data()  # Price - not returns

    # Compare portfolio to df with benchmarks
    metrics = backtester.performance_metrics(df, port_val, compare_assets=True)
    print(metrics)


    # Plotting
    df = df.iloc[-len(port_val):]

    save = False
    if save == True:
        metrics.round(4).to_latex(path + 'asset_performance.tex')
        plot_performance(df, port_val, weights, save=True)
    else:
        plot_performance(df, port_val, weights, save=False)








    # print('transaction costs:', (1-backtester.trans_cost).prod())
    # print('highest trans cost', backtester.trans_cost.max())