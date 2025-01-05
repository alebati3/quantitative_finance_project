import os
import sys
import numpy as np
import time

from w_k_means import w_k_means
from data_generator import generate_regimes, gbm_path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from pre_processing_utils import w_lift_function, data_par
from accuracy_utils import *
from plot_utils import *
from io_utils import *

# GBM parameters
gbm_par = np.array(
    [[0.02, 0.2], #mu,sigma *bull-regime*
    [-0.02, 0.3]]) #mu,sigma *bear-regime*

# MJD parameters
mjd_par = np.array(
    [[0.05, 0.2, 5, 0.02, 0.0125], # (mu,sigma, lambda, gamma, delta) bull-regime
    [-0.05, 0.4, 10, -0.04, 0.1]]) # (mu,sigma, lambda, gamma, delta) bear-regime


def synthetic_w_validation(path):

    # time horizon in years
    T = 20
    # number of hourly returns 
    N = int(T * 252 * 7)
    # lenght of regime change
    l_regime = int(0.5  * 252 * 7)
    # time interval
    dt = T / N
    # time evolution
    t = np.linspace(0, T, N+1)

    # regimes generation
    regimes, theo_return_labels, theo_price_labels = generate_regimes(N, l_regime)

    # clustering parameters
    h1 = 35
    h2 = 28

    p = 1
    max_iter = 400
    tol = 1e-6

    # actual variables
    N_prime, M = data_par(N, h1, h2)
    theo_return_labels = theo_return_labels[:N_prime]
    restricted_regimes = [sub[sub <= N_prime] for sub in regimes]
    regimes = np.array(restricted_regimes, dtype=object)

    if path == 'GBM':

        N_trials = 5 
        rofs = np.zeros(N_trials)
        rons = np.zeros(N_trials)
        ta = np.zeros(N_trials)
        iteration_times = np.zeros(N_trials)     

        for i in range(N_trials):
            # path generation
            prices = gbm_path(N, theo_price_labels, t, dt, gbm_par) 
            log_returns = np.diff(np.log(prices))
            print(np.mean(log_returns))
            start = time.time()
            log_returns = log_returns[: N_prime]
            # clustering pre processing
            X_wasserstein = w_lift_function(h1, h2, log_returns, M)
            # clustering implementation
            wkmeans, off_regime_index, on_regime_index = w_k_means(X_wasserstein, p, max_iter, tol)
            # accuracy scores
            r_counter = opt_counter(wkmeans, len(log_returns), M, h1, h2, o=False)
            rofs[i], rons[i], ta[i] = compute_accuracy_scores(r_counter, off_regime_index, on_regime_index, theo_return_labels)
            iteration_times[i] = time.time() - start

    # i need to save the results..

    elif path == 'MJD':


        N_trials = 5
        rofs = np.zeros(N_trials)
        rons = np.zeros(N_trials)
        ta = np.zeros(n_runs)
        iteration_times = np.zeros(n_runs)     

        for i in range(N_trials):
            # path generation
            prices = mjd_path(N, theo_price_labels, t, dt, mjd_par)
            log_returns = np.diff(np.log(prices))
            start = time.time()
            log_returns = log_returns[: N_prime]
            # clustering pre processing
            X_wasserstein = w_lift_function(h1, h2, log_returns, M)
            # clustering implementation
            wkmeans, off_regime_index, on_regime_index = w_k_means(X_wasserstein, p, max_iter, tol)
            # accuracy scores
            r_counter = opt_counter(wkmeans, len(log_returns), M, h1, h2, o=False)
            rofs[i], rons[i], ta[i] = compute_accuracy_scores(r_counter, off_regime_index, on_regime_index, theo_return_labels)
            iteration_times[i] = time.time() - start

        










if __name__ == "__main__":
    synthetic_w_validation('GBM')