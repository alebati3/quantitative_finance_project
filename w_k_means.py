import os
import sys
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import minkowski
from data_generator import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from pre_processing_utils import *
from accuracy_utils import *
from plot_utils import *
from io_utils import *


class WassersteinKMeans:
    def __init__(self, p, max_iter, tol, n_clusters = 2, random_state=None):
        self.n_clusters = n_clusters
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        # Obs: the rows of X are already ordered
        np.random.seed(self.random_state)
        # n_atoms represents the number of atoms for the empirical cdf
        n_samples, n_atoms = X.shape

        # Initialize cluster centers
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

        for i in range(self.max_iter):
            # Compute distances and assign clusters
            distances = pairwise_distances(X, self.cluster_centers_, metric='minkowski') / (n_atoms**(1/self.p))
            labels = np.argmin(distances, axis=1)

            # Compute new cluster centers
            new_centers = np.array([np.median(X[labels == j] ,axis=0) for j in range(self.n_clusters)])
            # just to be sure that the new centroids are ordered sequences
            new_centers.sort()
            
            # Check for convergence
            loss = 0
            for j in range(self.n_clusters):
                
                loss = loss + minkowski(self.cluster_centers_[j], new_centers[j], p=self.p) / (n_atoms**(1/self.p))
            if loss < self.tol:
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels
        return self

    def predict(self, X):
        distances = pairwise_distances(X, self.cluster_centers_, metric='minkowski') / (X.shape[1]**(1/self.p))
        return np.argmin(distances, axis=1)


def w_k_means(X, p, max_iter, tol, seed_clustering):
    
    wkmeans = WassersteinKMeans(p=p, max_iter=max_iter, tol=tol, random_state=seed_clustering)
    # Fit the Wasserstein KMeans
    wkmeans.fit(X)

    # off-regime-> cluster with a higher numeber of elements
    off_regime_index = 0 
    # on-regime-> cluster with a lower numeber of elements
    on_regime_index = 1 
    # check regime
    if (wkmeans.labels_ == 0).sum() < (wkmeans.labels_ == 1).sum():
        off_regime_index = 1
        on_regime_index = 0
        
    return wkmeans, off_regime_index, on_regime_index

def w_main():
    # for W k-means
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
                w_main1(path)
                # Logica per generare dati di IBM
            elif sub_scelta == 2:
                print("\nHai scelto di analizzare dati di MSFT.")
                path = 'MSFT'
                w_main1(path)
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
                w_main2(path)
                # Logica per analizzare dati sintetici di GBM
            elif sub_scelta == 2:
                print("\nHai scelto di analizzare MJD.")
                path = 'MJD'
                w_main2(path)
                # Logica per analizzare dati sintetici di GBM
            else:
                print("\nScelta non valida. Riprova.")

        else:
            print("\nScelta non valida. Riprova.")

    except ValueError:
        print("\nErrore: Inserisci un numero valido.")


def w_main1(path):
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

    X_wasserstein = w_lift_function(h1, h2, log_returns, M)
    print(f'number of sub sequences = {M}')

    # clustetring parameters
    parameters = ask_w_param()
    print(f"Selected parameters: {parameters}")

    p = parameters['p']
    clustering_seed = parameters['clustering_seed']
    max_iter = parameters['max_iter']
    tol = parameters['tol']

    # clustering implementation
    wkmeans, off_regime_index, on_regime_index = w_k_means(X_wasserstein, p, max_iter, tol, clustering_seed)

    # create directory
    directory_path = f'figures/{path}/W_k_means/h_{h1}_{h2}/max_iter_{max_iter}_tol_{tol}/clustering_seed_{clustering_seed}/p_{p}'
    ensure_directory_exists(directory_path)
    
    # projection of the clusters on the mu_std plane
    wk_mu_std_plot(X_wasserstein, wkmeans, off_regime_index, on_regime_index, directory_path)
    # projection of the clusters on the kurt_skew plane
    wk_kurt_skew_plot(X_wasserstein, wkmeans, off_regime_index, on_regime_index, directory_path)

    # plots classified real log returns and path price
    r_counter = opt_counter(wkmeans, len(log_returns), M, h1, h2, o=False)
    classified_real_log_returns_plot(r_counter, off_regime_index, log_returns, t, directory_path)
    classified_real_price_path_plot(r_counter, off_regime_index, prices, t, directory_path)

    # # projection of the clusters on the mu_skew plane
    # wk_mu_skew_plot(p, clustering_seed, h1, h2, path, X_wasserstein, wkmeans, max_iter, tol, off_regime_index, on_regime_index)
    # # projection of the clusters on the std_skew plane
    # wk_std_skew_plot(p, clustering_seed, h1, h2, path, X_wasserstein, wkmeans, max_iter, tol, off_regime_index, on_regime_index)
    # # projection of the clusters on the excess kurt_std plane
    # wk_kurt_std_plot(p, clustering_seed, h1, h2, path, X_wasserstein, wkmeans, max_iter, tol, off_regime_index, on_regime_index)
    # # projection of the clusters on the kurt_mu plane
    # wk_kurt_mu_plot(p, clustering_seed, h1, h2, path, X_wasserstein, wkmeans, max_iter, tol, off_regime_index, on_regime_index)




def w_main2(path):
    # just for W k-means; synthetic data

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

    X_wasserstein = w_lift_function(h1, h2, log_returns, M)
    print(f'number of sub sequences = {M}')


    # clustetring parameters
    parameters = ask_w_param()
    print(f"Selected parameters: {parameters}")

    p = parameters['p']
    clustering_seed = parameters['clustering_seed']
    max_iter = parameters['max_iter']
    tol = parameters['tol']
    
    # clustering implementation
    wkmeans, off_regime_index, on_regime_index = w_k_means(X_wasserstein, p, max_iter, tol, clustering_seed)

    # create directory 
    directory_path = f'figures/{path}/path_seed_{path_seed}/W_k_means/h_{h1}_{h2}/max_iter_{max_iter}_tol_{tol}/clustering_seed_{clustering_seed}/p_{p}'
    ensure_directory_exists(directory_path)    

    # projection of the clusters on the mu-sted plane
    wk_mu_std_plot(X_wasserstein, wkmeans, off_regime_index, on_regime_index, directory_path)
    # projection of the clusters on the excess kurtosis-skew plane
    wk_kurt_skew_plot(X_wasserstein, wkmeans, off_regime_index, on_regime_index, directory_path)

    # plots of classified synthetic log_returns and path price
    r_counter = opt_counter(wkmeans, len(log_returns), M, h1, h2, o=False)
    classified_synthetic_log_returns_plot(r_counter, off_regime_index, log_returns, t, regimes, directory_path)
    classified_synthetic_price_path_plot(r_counter, off_regime_index, prices, t, regimes, directory_path)

    # accuracy scores 
    ROFS, RONS, TA = compute_accuracy_scores(r_counter, off_regime_index, on_regime_index, theo_return_labels)
    save_and_print_values(directory_path, ROFS, RONS, TA, dec=2)





if __name__ == "__main__":  
    w_main() 


  