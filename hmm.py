import os
import sys
import numpy as np
from hmmlearn import hmm
from data_generator import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from pre_processing_utils import *
from accuracy_utils import *
from plot_utils import *
from io_utils import *


def hmm_clustering(log_returns, max_iter, tol, seed_clustering=None):

    # using log returns
    time_series_data = log_returns.reshape(-1, 1)

    # Define the HMM
    model = hmm.GaussianHMM(n_components=2, covariance_type='diag', random_state=seed_clustering, n_iter=max_iter, tol=tol)

    # Fit the HMM to the time series data
    model.fit(time_series_data)

    # Predict hidden states
    hmm_labels = model.predict(time_series_data)
    # Predict hidden states
    hmm_labels = model.predict(time_series_data)
    if model.monitor_.converged:
        print('convergence reached!')

    off_regime_index = 0
    on_regime_index = 1

    if (hmm_labels == 0).sum() < (hmm_labels == 1).sum():
        off_regime_index = 1
        on_regime_index = 0
        
    return hmm_labels, off_regime_index, on_regime_index

def hmm_main():
    # for Hidden Markov models
    """
    Funzione principale che gestisce l'interazione con l'utente.
    """
    print("Scegli i dati da analizzare con W k-means:")
    print("1 - Analizzare dati reali")
    print("2 - Analizzare dati sintetici")

    try:
        scelta = int(input("Inserisci il numero dell'opzione desiderata: "))

        if scelta == 1:
            print("\nHai scelto di analizzare dati reali.")
            print("Scegli un dataset da analizzare:")
            print("1 - IBM")
            print("2 - MSFT")

            sub_scelta = int(input("Inserisci il numero del dataset: "))

            if sub_scelta == 1:
                print("\nHai scelto di analizzare dati di IBM.")
                path = 'IBM'
                hmm_main1(path)
                # Logica per generare dati di IBM
            elif sub_scelta == 2:
                print("\nHai scelto di analizzare dati di MSFT.")
                path = 'MSFT'
                hmm_main1(path)
                # Logica per generare dati di MSFT
            else:
                print("\nScelta non valida. Riprova.")

        elif scelta == 2:
            print("\nHai scelto di analizzare dati sintetici.")
            print("Scegli la tipologia dei dati sintetici:")
            print("1 - Geometric Brownian Motion (GBM)")
            print("2 - Merton Jumpo Diffusion (MJD)")

            sub_scelta = int(input("Inserisci il numero del metodo: "))

            if sub_scelta == 1:
                print("\nHai scelto di analizzare GBM.")
                path = 'GBM'
                hmm_main2(path)
                # Logica per analizzare dati sintetici di GBM
            elif sub_scelta == 2:
                print("\nHai scelto di analizzare MJD.")
                path = 'MJD'
                hmm_main2(path)
                # Logica per analizzare dati sintetici di GBM
            else:
                print("\nScelta non valida. Riprova.")

        else:
            print("\nScelta non valida. Riprova.")

    except ValueError:
        print("\nErrore: Inserisci un numero valido.")


def hmm_main2(path):
    # just for HMM; synthetic paths
    # synthetic path generation
    path_information = synthetic_path_generation(path)

    prices = path_information['prices']
    log_returns = path_information['log_returns']
    t = path_information['t']
    regimes = path_information['regimes']
    theo_return_labels = path_information['theo_return_labels']
    path_seed = path_information['path_seed']

    # clustetring parameters
    parameters = ask_hmm_param() 
    print(f"Selected parameters: {parameters}")

    clustering_seed = parameters['clustering_seed']
    max_iter = parameters['max_iter']
    tol = parameters['tol']

    # hmm clustering
    hmm_labels, off_regime_index, on_regime_index = hmm_clustering(log_returns, clustering_seed, max_iter, tol)

    # create directory
    directory_path = f'figures/{path}/path_seed_{path_seed}/HMM/max_iter_{max_iter}_tol_{tol}/clustering_seed_{clustering_seed}'
    ensure_directory_exists(directory_path)    

    ROFS, RONS, TA = compute_accuracy_scores_hmm(hmm_labels, theo_return_labels, off_regime_index, on_regime_index)
    save_and_print_values(directory_path, ROFS, RONS, TA, dec=2)

    #plots:
    hmm_classified_synthetic_log_returns_plot(log_returns, t, hmm_labels, off_regime_index, regimes, directory_path)
    hmm_classified_synthetic_price_path_plot(prices, t, hmm_labels, off_regime_index, regimes, directory_path)


def hmm_main1(path):
    # path information
    path_information = import_real_data(path)

    prices = path_information['prices']
    log_returns = path_information['log_returns']
    t = path_information['t']


    # clustetring parameters
    parameters = ask_hmm_param()
    print(f"Selected parameters: {parameters}")

    clustering_seed = parameters['clustering_seed']
    max_iter = parameters['max_iter']
    tol = parameters['tol']

    # hmm clustering
    hmm_labels, off_regime_index, on_regime_index = hmm_clustering(log_returns, clustering_seed, max_iter, tol)

    # create directory
    directory_path = f'figures/{path}/HMM/max_iter_{max_iter}_tol_{tol}/clustering_seed_{clustering_seed}'
    ensure_directory_exists(directory_path)    


    hmm_classified_real_log_returns_plot(log_returns, t, hmm_labels, off_regime_index, directory_path)
    hmm_classified_real_price_path_plot(prices, t, hmm_labels, off_regime_index, directory_path)


if __name__ == "__main__":   
    hmm_main()
