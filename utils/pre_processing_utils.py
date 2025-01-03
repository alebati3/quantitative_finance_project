import numpy as np
from sklearn.preprocessing import StandardScaler


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
