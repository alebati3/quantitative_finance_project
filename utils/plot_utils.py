import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from io_utils import *
from accuracy_utils import compare_columns, hmm_compare_columns



def synthetic_price_path_plot(t, prices, regimes, directory_path):
    file_name = 'synthetic_path_price.pdf'
    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t,prices)

    # plot of regime changes
    for i in range(10):
        if i == 0:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3, label='regime changes')
            
        else:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3)
            
        
    plt.xlabel("time (years)")
    plt.ylabel("stock price")
    plt.grid()
    plt.legend()
    plt.title(file_path)
    plt.savefig(file_path)
    plt.show()


def synthetic_log_returns_plot(t, log_returns, subsequences, directory_path):
    file_name = 'synthetic_log_returns.pdf'
    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t[:-1], log_returns)

    # plot of regime changes
    for i in range(10):
        if i == 0:
            plt.axvspan(t[subsequences[i][0]], t[subsequences[i][-1]], color='red', alpha=0.3, label='regime changes')
            
        else:
            plt.axvspan(t[subsequences[i][0]], t[subsequences[i][-1]], color='red', alpha=0.3)
            
        
    plt.xlabel("time (years)")
    plt.ylabel("log returns")
    plt.grid()
    plt.legend()
    plt.title(file_path)
    plt.savefig(file_path)
    plt.show()


def real_price_path_plot(t, s, directory_path):
    file_name = 'path_price.pdf'
    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)

    plt.figure(figsize=(10, 6))
    plt.plot(t, s)
    plt.xlabel("time (years)")
    plt.ylabel("price")
    plt.grid()
    plt.title(file_path)
    plt.savefig(file_path)
    plt.show()


def real_log_returns_plot(t, r, directory_path):
    file_name = 'log_returns.pdf'
    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)

    plt.figure(figsize=(10, 6))
    plt.plot(t[:-1], r)
    plt.xlabel("time (years)")
    plt.ylabel("log returns")
    plt.grid()
    plt.title(file_path)
    plt.savefig(file_path)
    plt.show()


def wk_mu_std_plot(X, wkmeans, off_regime_index, on_regime_index, directory_path):

    # scatter plot of empirical cdf
    plt.figure(figsize=(10, 6))
    point_size = 4
    plt.scatter(
        np.std(X[wkmeans.labels_ == off_regime_index], axis=1),
        np.mean(X[wkmeans.labels_ == off_regime_index], axis=1),
        marker='.', color='green', alpha=0.5, s=point_size)
    plt.scatter(
        np.std(X[wkmeans.labels_ == on_regime_index], axis=1),
        np.mean(X[wkmeans.labels_ == on_regime_index], axis=1),  
        marker='.', color='orange', alpha=0.5, s=point_size)
    # scatter plot of centroids
    plt.scatter(np.std(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                np.mean(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                color='blue', marker='x', label='centroid 0')
    plt.scatter(np.std(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                np.mean(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                color='red', marker='x', label='centroid 1')

    plt.xlabel(f'$\sigma$', size=13)
    plt.ylabel(f'$\mu$', size=13)
    # paper notation
    plt.legend()
    # synthetic price path
    file_name = 'mu_std.pdf'
    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()

def wk_kurt_skew_plot(X, wkmeans, off_regime_index, on_regime_index, directory_path):
    # scatter plot of empirical cdf
    plt.figure(figsize=(10, 6))
    point_size = 4
    plt.scatter(
        skew(X[wkmeans.labels_ == off_regime_index], axis=1),
        kurtosis(X[wkmeans.labels_ == off_regime_index], axis=1),
        marker='.', color='green', alpha=0.5, s=point_size)
    plt.scatter(
        skew(X[wkmeans.labels_ == on_regime_index], axis=1),
        kurtosis(X[wkmeans.labels_ == on_regime_index], axis=1),  
        marker='.', color='orange', alpha=0.5, s=point_size)
    # scatter plot of centroids
    plt.scatter(skew(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                kurtosis(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                color='blue', marker='x', label='centroid 0')
    plt.scatter(skew(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                kurtosis(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                color='red', marker='x', label='centroid 1')

    plt.xlabel(f'skew', size=13)
    plt.ylabel(f'excess kurtosis', size=13)
    plt.legend()
    # Construct the full file path
    file_name = 'kurt_skew.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()

# It needs to be adjusted !
def wk_mu_skew_plot(p, seed_clustering, h_1, h_2, path, X, wkmeans, max_iter, tol, off_regime_index, on_regime_index, seed_path = None):

    # scatter plot of empirical cdf
    plt.figure(figsize=(10, 6))
    point_size = 4
    plt.scatter(
        skew(X[wkmeans.labels_ == off_regime_index], axis=1),
        np.mean(X[wkmeans.labels_ == off_regime_index], axis=1),
        marker='.', color='green', alpha=0.5, s=point_size)
    plt.scatter(
        skew(X[wkmeans.labels_ == on_regime_index], axis=1),
        np.mean(X[wkmeans.labels_ == on_regime_index], axis=1),  
        marker='.', color='orange', alpha=0.5, s=point_size)
    # scatter plot of centroids
    plt.scatter(skew(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                np.mean(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                color='blue', marker='x', label='centroid 0')
    plt.scatter(skew(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                np.mean(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                color='red', marker='x', label='centroid 1')

    plt.xlabel('skew', size=13)
    plt.ylabel(f'$\mu$', size=13)
    # paper notation
    plt.legend()
    if path == 'GBM' or path == 'MJD':
        # synthetic price path
        file_name = f'figures/{path}_{seed_path}_p_{p}_W_{seed_clustering}_h_{h_1}_{h_2}_ite_{max_iter}_tol_{tol}_mu_skew.pdf'
        plt.savefig(file_name, bbox_inches='tight')    

    else:
        # real price path
        file_name = f'figures/{path}_p_{p}_W_{seed_clustering}_h_{h_1}_{h_2}_ite_{max_iter}_tol_{tol}_mu_skew.pdf'
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()

# It needs to be adjusted !
def wk_std_skew_plot(p, seed_clustering, h_1, h_2, path, X, wkmeans, max_iter, tol, off_regime_index, on_regime_index, seed_path = None):

    # scatter plot of empirical cdf
    plt.figure(figsize=(10, 6))
    point_size = 4
    plt.scatter(
        skew(X[wkmeans.labels_ == off_regime_index], axis=1),
        np.std(X[wkmeans.labels_ == off_regime_index], axis=1),
        marker='.', color='green', alpha=0.5, s=point_size)
    plt.scatter(
        skew(X[wkmeans.labels_ == on_regime_index], axis=1),
        np.std(X[wkmeans.labels_ == on_regime_index], axis=1),  
        marker='.', color='orange', alpha=0.5, s=point_size)
    # scatter plot of centroids
    plt.scatter(skew(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                np.std(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                color='blue', marker='x', label='centroid 0')
    plt.scatter(skew(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                np.std(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                color='red', marker='x', label='centroid 1')

    plt.xlabel('skew', size=13)
    plt.ylabel(f'$\sigma$', size=13)
    # paper notation
    plt.legend()
    if path == 'GBM' or path == 'MJD':
        # synthetic price path
        file_name = f'figures/{path}_{seed_path}_p_{p}_W_{seed_clustering}_h_{h_1}_{h_2}_ite_{max_iter}_tol_{tol}_std_skew.pdf'
        plt.savefig(file_name, bbox_inches='tight')    

    else:
        # real price path
        file_name = f'figures/{path}_p_{p}_W_{seed_clustering}_h_{h_1}_{h_2}_ite_{max_iter}_tol_{tol}_std_skew.pdf'
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()

# It needs to be adjusted !
def wk_kurt_std_plot(p, seed_clustering, h_1, h_2, path, X, wkmeans, max_iter, tol, off_regime_index, on_regime_index, seed_path = None):
    # scatter plot of empirical cdf
    plt.figure(figsize=(10, 6))
    point_size = 4
    plt.scatter(
        np.std(X[wkmeans.labels_ == off_regime_index], axis=1),
        kurtosis(X[wkmeans.labels_ == off_regime_index], axis=1),
        marker='.', color='green', alpha=0.5, s=point_size)
    plt.scatter(
        np.std(X[wkmeans.labels_ == on_regime_index], axis=1),
        kurtosis(X[wkmeans.labels_ == on_regime_index], axis=1),  
        marker='.', color='orange', alpha=0.5, s=point_size)
    # scatter plot of centroids
    plt.scatter(np.std(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                kurtosis(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                color='blue', marker='x', label='centroid 0')
    plt.scatter(np.std(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                kurtosis(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                color='red', marker='x', label='centroid 1')

    plt.xlabel(f'$\sigma$', size=13)
    plt.ylabel(f'excess kurtosis', size=13)
    plt.legend()
    if path == 'GBM' or path == 'MJD':
        # synthetic price path
        file_name = f'figures/{path}_{seed_path}_p_{p}_W_{seed_clustering}_h_{h_1}_{h_2}_ite_{max_iter}_tol_{tol}_kurt_std.pdf'
        plt.savefig(file_name, bbox_inches='tight')    

    else:
        # real price path
        file_name = f'figures/{path}_p_{p}_W_{seed_clustering}_h_{h_1}_{h_2}_ite_{max_iter}_tol_{tol}_kurt_std.pdf'
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()

# It needs to be adjusted !
def wk_kurt_mu_plot(p, seed_clustering, h_1, h_2, path, X, wkmeans, max_iter, tol, off_regime_index, on_regime_index, seed_path = None):
    # scatter plot of empirical cdf
    plt.figure(figsize=(10, 6))
    point_size = 4
    plt.scatter(
        np.mean(X[wkmeans.labels_ == off_regime_index], axis=1),
        kurtosis(X[wkmeans.labels_ == off_regime_index], axis=1),
        marker='.', color='green', alpha=0.5, s=point_size)
    plt.scatter(
        np.mean(X[wkmeans.labels_ == on_regime_index], axis=1),
        kurtosis(X[wkmeans.labels_ == on_regime_index], axis=1),  
        marker='.', color='orange', alpha=0.5, s=point_size)
    # scatter plot of centroids
    plt.scatter(np.mean(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                kurtosis(wkmeans.cluster_centers_, axis=1)[off_regime_index],
                color='blue', marker='x', label='centroid 0')
    plt.scatter(np.mean(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                kurtosis(wkmeans.cluster_centers_, axis=1)[on_regime_index],
                color='red', marker='x', label='centroid 1')

    plt.xlabel(f'$\mu$', size=13)
    plt.ylabel(f'excess kurtosis', size=13)
    plt.legend()
    if path == 'GBM' or path == 'MJD':
        # synthetic price path
        file_name = f'figures/{path}_{seed_path}_p_{p}_W_{seed_clustering}_h_{h_1}_{h_2}_ite_{max_iter}_tol_{tol}_kurt_mu.pdf'
        plt.savefig(file_name, bbox_inches='tight')    

    else:
        # real price path
        file_name = f'figures/{path}_p_{p}_W_{seed_clustering}_h_{h_1}_{h_2}_ite_{max_iter}_tol_{tol}_kurt_mu.pdf'
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()




def skewness_and_kurtosis(M):
    """
    Calculate skewness and excess kurtosis using raw moments.
    
    Parameters:
    M1: First raw moment (mean)
    M2: Second raw moment (variance-related)
    M3: Third raw moment
    M4: Fourth raw moment
    
    Returns:
    skewness, excess kurtosis
    """
    M1 = M[0]
    M2 = M[1]
    M3 = M[2]
    M4 = M[3]
    
    # Calculate variance (second central moment, which is just variance)
    mu2 = M2 - M1**2

    # Calculate third central moment
    mu3 = M3 - 3 * M1 * M2 + 2 * M1**3

    # Calculate fourth central moment
    mu4 = M4 - 4 * M1 * M3 + 6 * M1**2 * M2 - 3 * M1**4

    # Calculate skewness
    skewness = mu3 / mu2**(3/2)

    # Calculate excess kurtosis (subtract 3 from kurtosis)
    excess_kurtosis = (mu4 / mu2**2) - 3

    return skewness, excess_kurtosis

def mk_mu_std_plot(lift_matrix, scaler, mkmeans, off_regime_index, on_regime_index, directory_path):
    # centroids in the real space
    centroids = scaler.inverse_transform(mkmeans.cluster_centers_) 

    # scatter plot of empirical cdf
    plt.figure(figsize=(10, 6))
    point_size = 4
    plt.scatter(
        np.std(lift_matrix[mkmeans.labels_ == off_regime_index], axis=1),
        np.mean(lift_matrix[mkmeans.labels_ == off_regime_index], axis=1),
        marker='.', color='green', alpha=0.3, s=point_size)
    plt.scatter(
        np.std(lift_matrix[mkmeans.labels_ == on_regime_index], axis=1),
        np.mean(lift_matrix[mkmeans.labels_ == on_regime_index], axis=1),  
        marker='.', color='orange', alpha=1, s=point_size)

    # scatter plot of centroids

    plt.scatter(np.sqrt(centroids[off_regime_index][1] - (centroids[off_regime_index][0])**2),
                centroids[off_regime_index][0],
                color='blue', marker='x', label='centroid 0')
    plt.scatter(np.sqrt(centroids[on_regime_index][1] - (centroids[on_regime_index][0])**2),
                centroids[on_regime_index][0],
                color='red', marker='x', label='centroid 1')

    plt.xlabel(f'$\sigma$', size=13)
    plt.ylabel(f'$\mu$', size=13)
    # paper notation
    plt.legend()
    # synthetic price path
    file_name = 'mu_std.pdf'
    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()


def mk_kurt_skew_plot(lift_matrix, scaler, mkmeans, off_regime_index, on_regime_index, directory_path):
    # centroids in the real space
    centroids = scaler.inverse_transform(mkmeans.cluster_centers_) 
    # scatter plot of empirical cdf
    plt.figure(figsize=(10, 6))
    point_size = 4
    plt.scatter(
        skew(lift_matrix[mkmeans.labels_ == off_regime_index], axis=1),
        kurtosis(lift_matrix[mkmeans.labels_ == off_regime_index], axis=1),
        marker='.', color='green', alpha=0.3, s=point_size)
    plt.scatter(
        skew(lift_matrix[mkmeans.labels_ == on_regime_index], axis=1),
        kurtosis(lift_matrix[mkmeans.labels_ == on_regime_index], axis=1),  
        marker='.', color='orange', alpha=0.4, s=point_size)
    # scatter plot of centroids
    skewness_0, excess_kurtosis_0 = skewness_and_kurtosis(centroids[off_regime_index])
    skewness_1, excess_kurtosis_1 = skewness_and_kurtosis(centroids[on_regime_index])
    plt.scatter(skewness_0,
                excess_kurtosis_0,
                color='blue', marker='x', label='centroid 0')

    plt.scatter(skewness_1,
                excess_kurtosis_1,
                color='red', marker='x', label='centroid 1')


    plt.xlabel(f'skew', size=13)
    plt.ylabel(f'excess kurtosis', size=13)
    plt.legend()
    # Construct the full file path
    file_name = 'kurt_skew.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()




def classified_synthetic_log_returns_plot(r_counter, off_regime_index, log_returns, t, regimes, directory_path):

    b = compare_columns(r_counter, off_regime_index)
    color = ['green', 'red', 'blue']
    start_j = 0
    end_j = 0
    m_size = 1

    if not 2 in b:
        print('No ambiguities in classified synthetic log returns')
    else:
        print('Ambiguities in classified synthetic log returns')
        
    plt.figure(figsize=(10, 6))
    for i in range(0, len(log_returns)):
        
        if i == (len(log_returns) - 1):
            plt.plot(t[start_j: end_j + 1], log_returns[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
        
        elif b[i] == b[i+1]:
            end_j = i + 1
            
        else:
            plt.plot(t[start_j: end_j + 1], log_returns[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
            start_j = i + 1
            end_j = i + 1
            
    for i in range(10):
        if i == 0:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3, label='regime changes')
            
        else:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3)        

    plt.legend()  
    plt.ylabel('log-returns')
    plt.xlabel('time (years)')
     # Construct the full file path
    file_name = 'classified_synthetic_log_returns.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()



def classified_real_log_returns_plot(r_counter, off_regime_index, log_returns, t, directory_path):

    b = compare_columns(r_counter, off_regime_index)
    color = ['green', 'red', 'blue']
    start_j = 0
    end_j = 0
    m_size = 1

    if not 2 in b:
        print('No ambiguities in classified real log returns')
    else:
        print('Ambiguities in classified real log returns')
        
    plt.figure(figsize=(10, 6))
    for i in range(0, len(log_returns)):
        
        if i == (len(log_returns) - 1):
            plt.plot(t[start_j: end_j + 1], log_returns[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
        
        elif b[i] == b[i+1]:
            end_j = i + 1
            
        else:
            plt.plot(t[start_j: end_j + 1], log_returns[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
            start_j = i + 1
            end_j = i + 1
                    

    plt.ylabel('log-returns')
    plt.xlabel('time (years)')
     # Construct the full file path
    file_name = 'classified_real_log_returns.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()



# It could be modified ...
def classified_synthetic_price_path_plot(r_counter, off_regime_index, prices, t, regimes, directory_path):
    log_returns = np.diff(np.log(prices))
    b = compare_columns(r_counter, off_regime_index)
    color = ['green', 'red', 'blue']
    start_j = 1
    end_j = 1
    m_size = 0.5

    if not 2 in b:
        print('No ambiguities in classified synthetic price path')
    else:
        print('Ambiguities in classified synthetic price path')
        
    plt.figure(figsize=(10, 6))
    for i in range(0, len(log_returns)):
        
        if i == (len(log_returns) - 1):
            plt.plot(t[start_j: end_j + 1], prices[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
        
        elif b[i] == b[i+1]:
            end_j = i + 2
            
        else:
            plt.plot(t[start_j: end_j + 1], prices[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
            start_j = i + 2
            end_j = i + 2
            
    for i in range(10):
        if i == 0:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3, label='regime changes')
            
        else:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3)        
        
            
    plt.legend()  
    plt.ylabel('price')
    plt.xlabel('time (years)')
     # Construct the full file path
    file_name = 'classified_synthetic_price_path.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()



# It could be modified ...
def classified_real_price_path_plot(r_counter, off_regime_index, prices, t, directory_path):
    log_returns = np.diff(np.log(prices))
    b = compare_columns(r_counter, off_regime_index)
    color = ['green', 'red', 'blue']
    start_j = 1
    end_j = 1
    m_size = 0.5

    if not 2 in b:
        print('No ambiguities in classified real price path')
    else:
        print('Ambiguities in classified real price path')
        
    plt.figure(figsize=(10, 6))
    for i in range(0, len(log_returns)):
        
        if i == (len(log_returns) - 1):
            plt.plot(t[start_j: end_j + 1], prices[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
        
        elif b[i] == b[i+1]:
            end_j = i + 2
            
        else:
            plt.plot(t[start_j: end_j + 1], prices[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
            start_j = i + 2
            end_j = i + 2
               
        
    plt.ylabel('price')
    plt.xlabel('time (years)')
     # Construct the full file path
    file_name = 'classified_real_price_path.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()

# HMM

def hmm_classified_synthetic_log_returns_plot(log_returns, t, hmm_labels, off_regime_index, regimes, directory_path):
    b = hmm_compare_columns(hmm_labels, off_regime_index)
    color = ['green', 'red']
    # set the size of line and marker
    m_size = 0.5

    start_j = 0
    end_j = 0

    plt.figure(figsize=(10, 6))
    for i in range(0, len(log_returns)):
        
        if i == (len(log_returns) - 1):
            plt.plot(t[start_j: end_j + 1], log_returns[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
        
        elif b[i] == b[i+1]:
            end_j = i + 1
            
        else:
            plt.plot(t[start_j: end_j + 1], log_returns[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
            start_j = i + 1
            end_j = i + 1
            
    for i in range(10):
        if i == 0:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3, label='regime changes')
            
        else:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3)        
            
    plt.legend(loc='upper right')
    plt.ylabel('log-returns')
    plt.xlabel('time (years)')
     # Construct the full file path
    file_name = 'classified_synthetic_log_returns.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()

def hmm_classified_real_log_returns_plot(log_returns, t, hmm_labels, off_regime_index, directory_path):
    b = hmm_compare_columns(hmm_labels, off_regime_index)
    color = ['green', 'red']
    # set the size of line and marker
    m_size = 0.5

    start_j = 0
    end_j = 0

    plt.figure(figsize=(10, 6))
    for i in range(0, len(log_returns)):
        
        if i == (len(log_returns) - 1):
            plt.plot(t[start_j: end_j + 1], log_returns[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
        
        elif b[i] == b[i+1]:
            end_j = i + 1
            
        else:
            plt.plot(t[start_j: end_j + 1], log_returns[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
            start_j = i + 1
            end_j = i + 1
                   
            
    plt.ylabel('log-returns')
    plt.xlabel('time (years)')
     # Construct the full file path
    file_name = 'classified_real_log_returns.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()



def hmm_classified_synthetic_price_path_plot(prices, t, hmm_labels, off_regime_index, regimes, directory_path):
    log_returns = np.diff(np.log(prices))
    b = hmm_compare_columns(hmm_labels, off_regime_index)
    color = ['green', 'red']
    # set the size of line and marker
    m_size = 0.5

    start_j = 0
    end_j = 0

    plt.figure(figsize=(10, 6))
    for i in range(0, len(log_returns)):
        
        if i == (len(log_returns) - 1):
            plt.plot(t[start_j: end_j + 1], prices[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
        
        elif b[i] == b[i+1]:
            end_j = i + 2
            
        else:
            plt.plot(t[start_j: end_j + 1], prices[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
            start_j = i + 2
            end_j = i + 2
            
    for i in range(10):
        if i == 0:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3, label='regime changes')
            
        else:
            plt.axvspan(t[regimes[i][0]], t[regimes[i][-1]], color='red', alpha=0.3)        
        
            
    plt.legend(loc='upper right') 
    plt.ylabel('price')
    plt.xlabel('time (years)')
     # Construct the full file path
    file_name = 'classified_synthetic_price_path.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()



def hmm_classified_real_price_path_plot(prices, t, hmm_labels, off_regime_index, directory_path):
    log_returns = np.diff(np.log(prices))
    b = hmm_compare_columns(hmm_labels, off_regime_index)
    color = ['green', 'red']
    # set the size of line and marker
    m_size = 0.5

    start_j = 0
    end_j = 0

    plt.figure(figsize=(10, 6))
    for i in range(0, len(log_returns)):
        
        if i == (len(log_returns) - 1):
            plt.plot(t[start_j: end_j + 1], prices[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
        
        elif b[i] == b[i+1]:
            end_j = i + 2
            
        else:
            plt.plot(t[start_j: end_j + 1], prices[start_j: end_j + 1], 
                    color=color[b[i]], marker='.', linewidth=m_size, markersize=m_size)
            start_j = i + 2
            end_j = i + 2
        
              
    plt.ylabel('price')
    plt.xlabel('time (years)')
     # Construct the full file path
    file_name = 'classified_real_price_path.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.title(file_path)
    plt.savefig(file_path, bbox_inches='tight') 
    plt.show()