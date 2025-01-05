import numpy as np
from sklearn.preprocessing import StandardScaler



def data_par(n, h_1, h_2):
    '''
    Given the hyper parameters h_1 and h_2 it returns the number of sub-sequences M and the effective number of log-returns that
    are involved in the analysis N_prime. 
    n is the total number of log returns
    
    '''
    
    # check the number of possible sub sequences M
    i = 0
    while ((h_1 - h_2) * i + h_1) <= (n-1):
        i = i + 1

    # IMPORTANT parameters
    M = i 
    N_prime = (h_1 - h_2) * (M-1) + h_1 + 1
    
    return N_prime, M


def synthetic_path_processing(path_information, h1, h2):

    prices = path_information['prices']
    log_returns = path_information['log_returns']
    t = path_information['t']
    regimes = path_information['regimes']
    theo_return_labels = path_information['theo_return_labels']
    path_seed = path_information['path_seed'] # it won't be changed
    N = len(log_returns)

    N_prime, M = data_par(N, h1, h2)
    # actual variables
    prices = prices[: N_prime + 1]
    t = t[: N_prime + 1]

    log_returns = log_returns[: N_prime]

    theo_return_labels = theo_return_labels[:N_prime]

    restricted_regimes = [sub[sub <= N_prime] for sub in regimes]
    restricted_regimes = np.array(restricted_regimes, dtype=object)

    actual_path_information = {'prices': prices, 'log_returns': log_returns, 't': t,
        'regimes' : restricted_regimes, 'theo_return_labels': theo_return_labels, 'path_seed': path_seed}

    return M, actual_path_information


def real_path_processing(path_information, h1, h2):

    prices = path_information['prices']
    log_returns = path_information['log_returns']
    t = path_information['t']
    N = len(log_returns)

    N_prime, M = data_par(N, h1, h2)
    # actual variables
    prices = prices[: N_prime + 1]
    t = t[: N_prime + 1]
    log_returns = log_returns[: N_prime]

    actual_path_information = {'prices': prices, 'log_returns': log_returns, 't': t}

    return M, actual_path_information




def w_lift_function(h_1, h_2, log_returns, M):
    '''
    It returns a matrix (and the sorted version) in which the rows are the subsequences.
    
    '''

    # creation of the sub-sequences
    lift_matrix = np.ndarray((M, h_1 + 1))

    for j in range(0, M):
        lift_matrix[j] = log_returns[(h_1 - h_2) * j : (h_1 - h_2) * j + h_1 + 1]

    sorted_lift_matrix = np.sort(lift_matrix)
    return sorted_lift_matrix

def m_lift_function(h_1, h_2, log_returns, M):
    '''
    It returns a matrix (and the sorted version) in which the rows are the subsequences.
    
    '''

    # creation of the sub-sequences
    lift_matrix = np.ndarray((M, h_1 + 1))

    for j in range(0, M):
        lift_matrix[j] = log_returns[(h_1 - h_2) * j : (h_1 - h_2) * j + h_1 + 1]

    return lift_matrix 

# Function to compute the k-th raw moment along a specified axis
def raw_moment_nd(values, k, axis=None):
    return np.mean(values**k, axis=axis)


def moments_pre_processing(q, lift_matrix, o=True):
    # compute raw moments along the specified axis (axis=None computes the raw moments over the entire array)
    X_moments = np.array([raw_moment_nd(lift_matrix, k, axis=1) for k in range(1, q+1)]).T

    # initialize the StandardScaler
    scaler = StandardScaler()

    # fit and transform the data
    standardized_X_moments = scaler.fit_transform(X_moments)

    if o:
        return standardized_X_moments, scaler
    else:
        return standardized_X_moments
