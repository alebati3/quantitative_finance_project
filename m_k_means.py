import sys
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from data_generator import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from pre_processing_utils import *
from accuracy_utils import *
from plot_utils import *
from io_utils import *



class MKMeans:
    
    def __init__(self, max_iter, tol, n_clusters=2, random_state=None):
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)

        n_samples = X.shape[0]

        # Initialize cluster centers
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

        for i in range(self.max_iter):
            
            # Compute distances and assign clusters
            distances = pairwise_distances(X, self.cluster_centers_, metric='euclidean')
            labels = np.argmin(distances, axis=1)

            # Compute new cluster centers
            new_centers = np.array([np.mean(X[labels == j] ,axis=0) for j in range(self.n_clusters)])
            
            # Check for convergence
            loss = 0
            for j in range(self.n_clusters):
                loss = loss + np.linalg.norm(self.cluster_centers_[j] - new_centers[j])
                
            if loss < self.tol:
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels
        return self

    def predict(self, X):
        distances = pairwise_distances(X, self.cluster_centers_, metric=euclidean_distance)
        return np.argmin(distances, axis=1)
    

def m_k_means(X, max_iter, tol, seed_clustering=None):
    
    mkmeans = MKMeans(max_iter=max_iter, tol=tol, random_state=seed_clustering)
    # Fit the Wasserstein KMeans
    mkmeans.fit(X)

    # off-regime-> cluster with a higher numeber of elements
    off_regime_index = 0 
    # on-regime-> cluster with a lower numeber of elements
    on_regime_index = 1 
    # check regime
    if (mkmeans.labels_ == 0).sum() < (mkmeans.labels_ == 1).sum():
        off_regime_index = 1
        on_regime_index = 0
        
    return mkmeans, off_regime_index, on_regime_index

def m_main():
    # for M k-means
    """
    Funzione principale che gestisce l'interazione con l'utente.
    """
    print("Scegli i dati da analizzare con M k-means:")
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
                m_main1(path)
                # Logica per generare dati di IBM
            elif sub_scelta == 2:
                print("\nHai scelto di analizzare dati di MSFT.")
                path = 'MSFT'
                m_main1(path)
                # Logica per generare dati di MSFT
            else:
                print("\nScelta non valida. Riprova.")

        elif scelta == 2:
            print("\nHai scelto di analizzare dati sintetici.")
            print("Scegli i dati da generare e analizzare:")
            print("1 - Geometric Brownian Motion (GBM)")
            print("2 - Merton Jumpo Diffusion (MJD)")

            sub_scelta = int(input("Inserisci il numero del metodo: "))

            if sub_scelta == 1:
                print("\nHai scelto di analizzare GBM.")
                path = 'GBM'
                m_main2(path)
                # Logica per analizzare dati sintetici di GBM
            elif sub_scelta == 2:
                print("\nHai scelto di analizzare MJD.")
                path = 'MJD'
                m_main2(path)
                # Logica per analizzare dati sintetici di GBM
            else:
                print("\nScelta non valida. Riprova.")

        else:
            print("\nScelta non valida. Riprova.")

    except ValueError:
        print("\nErrore: Inserisci un numero valido.")



def m_main1(path):
    # just for W k-means; real data

    # path information
    path_information = import_real_data(path)

    # clustering pre processing
    h1, h2 = ask_h1_h2()
    print(f"Valori scelti: h1 = {h1}, h2 = {h2}")

    M, actual_path_information = real_path_processing(path_information, h1, h2)
    prices = actual_path_information['prices']
    log_returns = actual_path_information['log_returns']
    t = actual_path_information['t']

    # clustering pre processing
    lift_matrix = m_lift_function(h1, h2, log_returns, M)
    print(f'number of sub sequences = {M}')

    # clustetring parameters
    parameters = ask_m_param()
    print(f"Selected parameters: {parameters}")
    p = parameters['p']
    clustering_seed = parameters['clustering_seed']
    max_iter = parameters['max_iter']
    tol = parameters['tol']

    standardized_X_moments, scaler = moments_pre_processing(p, lift_matrix)
    # print the standardized data
    print(np.mean(standardized_X_moments, axis=0))
    print(np.std(standardized_X_moments, axis=0))

    # m k-means clustetring
    mkmeans, off_regime_index, on_regime_index = m_k_means(standardized_X_moments, max_iter, tol, clustering_seed)  

    # create directory
    directory_path = f'figures/{path}/M_k_means/h_{h1}_{h2}/max_iter_{max_iter}_tol_{tol}/clustering_seed_{clustering_seed}/p_{p}'
    ensure_directory_exists(directory_path)

    # projection of the clusters on the mu-sted plane
    mk_mu_std_plot(lift_matrix, scaler, mkmeans, off_regime_index, on_regime_index, directory_path)
    # projection of the clusters on the mu-sted plane
    if p > 3:
        mk_kurt_skew_plot(lift_matrix, scaler, mkmeans, off_regime_index, on_regime_index, directory_path)

    # plots classified real log returns and path price
    r_counter = opt_counter(mkmeans, len(log_returns), M, h1, h2, o=False)
    classified_real_log_returns_plot(r_counter, off_regime_index, log_returns, t, directory_path)
    classified_real_price_path_plot(r_counter, off_regime_index, prices, t, directory_path)



def m_main2(path):
    # just for M k-means; synthetic data

    # synthetic path generation
    path_information = synthetic_path_generation(path)

    # clustering pre processing
    h1, h2 = ask_h1_h2()
    print(f"Valori scelti: h1 = {h1}, h2 = {h2}")

    # actual variables
    M, actual_path_information = synthetic_path_processing(path_information, h1, h2)

    prices = actual_path_information['prices']
    log_returns = actual_path_information['log_returns']
    t = actual_path_information['t']
    regimes = actual_path_information['regimes']
    theo_return_labels = actual_path_information['theo_return_labels']
    path_seed = actual_path_information['path_seed'] 

    # clustering pre processing
    lift_matrix = m_lift_function(h1, h2, log_returns, M)
    print(f'number of sub sequences = {M}')


    # clustetring parameters
    parameters = ask_m_param()
    print(f"Selected parameters: {parameters}")
    p = parameters['p']
    clustering_seed = parameters['clustering_seed']
    max_iter = parameters['max_iter']
    tol = parameters['tol']

    standardized_X_moments, scaler = moments_pre_processing(p, lift_matrix)
    # print the standardized data
    print(np.mean(standardized_X_moments, axis=0))
    print(np.std(standardized_X_moments, axis=0))

    # m k-means clustetring
    mkmeans, off_regime_index, on_regime_index = m_k_means(standardized_X_moments, max_iter, tol, clustering_seed)  

    # scatter plots:
    # create directory
    directory_path = f'figures/{path}/path_seed_{path_seed}/M_k_means/h_{h1}_{h2}/max_iter_{max_iter}_tol_{tol}/clustering_seed_{clustering_seed}/p_{p}'
    ensure_directory_exists(directory_path)

    # projection of the clusters on the mu-sted plane
    mk_mu_std_plot(lift_matrix, scaler, mkmeans, off_regime_index, on_regime_index, directory_path)
    # projection of the clusters on the mu-sted plane
    if p > 3:
        mk_kurt_skew_plot(lift_matrix, scaler, mkmeans, off_regime_index, on_regime_index, directory_path)


    # accuracy scores
    r_counter = opt_counter(mkmeans, len(log_returns), M, h1, h2, o=False)
    classified_synthetic_log_returns_plot(r_counter, off_regime_index, log_returns, t, regimes, directory_path)
    classified_synthetic_price_path_plot(r_counter, off_regime_index, prices, t, regimes, directory_path)
    ROFS, RONS, TA = compute_accuracy_scores(r_counter, off_regime_index, on_regime_index, theo_return_labels)
    save_and_print_values(directory_path, ROFS, RONS, TA, dec=2)



if __name__ == "__main__": 

    m_main()  



