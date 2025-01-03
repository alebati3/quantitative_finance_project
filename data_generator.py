import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from pre_processing_utils import *
from accuracy_utils import *
from plot_utils import *
from io_utils import *

# time horizon in years
T = 20

# number of time steps
N = int(T * 252 * 7)

# change remige's lenght
l_regime = int(0.5  * 252 * 7)

# time interval
dt = T / N

# array of all the timesteps
timestep = np.linspace(0, T, N)

# GBM parameters
gbm_par = np.array(
    [[0.02, 0.2], #mu,sigma *bull-regime*
    [-0.02, 0.3]]) #mu,sigma *bear-regime*

mjd_par = np.array(
    [[0.05, 0.2, 5, 0.02, 0.0125], # (mu,sigma, lambda, gamma, delta) bull-regime
    [-0.05, 0.4, 10, -0.04, 0.1]]) # (mu,sigma, lambda, gamma, delta) bear-regime


def data_par(n, h_1, h_2):
    '''
    Given the hyper parameters h_1 and h_2 it returns the number of sub-sequences M and the effective number of log-returns that
    are involved in the analysis (N_prime).
    
    '''
    
    # check the number of possible sub sequences M
    i = 0
    # N - 2 (-1:from price to log-return and -1:becuase the last index is lenght of the array -1)
    while ((h_1 - h_2) * i + h_1) <= (n-2):
        i = i + 1

    # IMPORTANT parameters
    m = i 
    N_prime = (h_1 - h_2) * (m-1) + h_1 + 1
    
    return N_prime, m

# Funzione per generare un indice di partenza valido
def generate_start_index(A, subseq_length, used_indices, random_state=17):
    """
    Genera un indice di partenza casuale per sottosequenze non sovrapposte.

    Parameters:
    - A: array di input da cui generare gli indici
    - subseq_length: lunghezza della sottosequenza
    - used_indices: set di indici già usati
    - random_state: seed per la generazione casuale (default 17)

    Returns:
    - start_index: indice di partenza valido
    """
    np.random.seed(random_state)
    while True:
        # Genera un indice di partenza casuale
        start_index = np.random.randint(0, len(A) - subseq_length - 1)
        # Controlla se l'indice di partenza e l'indice finale (con buffer di 1) sono validi
        if all((start_index + i) not in used_indices for i in range(subseq_length + 1)):
            for i in range(subseq_length + 1):
                used_indices.add(start_index + i)
            return start_index

# Funzione principale
def generate_regimes(N_prime, subseq_length, num_subsequences=10):
    """
    Genera casualmente 10 intervalli di tempo distinti della stessa lunghezza.

    Parameters:
    - N_prime: dimensione dell'array di input
    - subseq_length: lunghezza delle sottosequenze
    - num_subsequences: numero di sottosequenze da generare (default 10)

    Returns:
    - subsequences: lista di sottosequenze generate
    - B: etichette per i log-returns
    - C: etichette per i prezzi
    """
    A = np.arange(0, N_prime+1)

    # Set per memorizzare gli indici di partenza usati
    used_indices = set()

    # Generazione delle sottosequenze random non sovrapposte con almeno un elemento di distanza
    subsequences = []
    for _ in range(num_subsequences):
        start_index = generate_start_index(A, subseq_length, used_indices)
        subsequences.append(A[start_index:start_index + subseq_length])

    subsequences = np.sort(np.array(subsequences), axis=0)
    
    # Label per i log-returns
    B = np.zeros(N_prime)
    for sub in subsequences:
        B[sub[0]: sub[-1]] = 1    
    B = B.astype(int)

    # Label per i prezzi
    C = np.zeros(N_prime+1)
    for sub in subsequences:
        C[sub] = 1    
    C = C.astype(int)

    return subsequences, B, C

def gbm(S0, mu, sigma, n, dt):
    """
    Simulates a Geometric Brownian Motion (GBM) by using the analytical solution.

    Parameters:
    S0 (float): Initial stock price
    mu (float): Drift coefficient
    sigma (float): Volatility coefficient
    T (float): Time horizon
    n (int): Number of time steps

    Returns:
    np.ndarray: Simulated stock prices

    """
    t = np.arange(1, n) * dt
    W = np.random.standard_normal(size=n-1) 
    W = np.cumsum(W) * np.sqrt(dt) # cumulative sum to simulate the Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = np.zeros(n)
    S[0] = S0
    S[1:] = S0 * np.exp(X)
    return S

def gbm_path(N_prime, C, t, dt, gbm_par, seed_path=None):
    '''
    It simulates the entire path of a GBM with regimes switch.
    
    '''
    np.random.seed(seed_path)
    # array of prices
    s = np.zeros(N_prime + 1)
    # initial stock price
    s[0] = 1
    s_0 = s[0]
    start_index = 0
    stop_index = 1

    for k in range(1, N_prime+1):
        if k == N_prime:
            s[start_index : stop_index + 1] = gbm(s_0, gbm_par[C[k]][0], gbm_par[C[k]][1], len(t[start_index : stop_index + 1]), dt)

        elif C[k] == C[k+1]:
            stop_index = k+1

        else:
            s[start_index : stop_index + 1] = gbm(s_0, gbm_par[C[k]][0], gbm_par[C[k]][1], len(t[start_index : stop_index + 1]), dt)
            #updates
            start_index = k
            s_0 = s[k]
            stop_index = k + 1
            
    return s


def mjd(S0, mu, sigma, lam, gamma, delta, n, dt):
    """
    Simulates a Merton Jump Diffusion process (MJD).

    Parameters:
    S0 (float): Initial stock price
    mu (float): Drift
    sigma (float): Volatility
    lambda_ (float): Jump intensity (average number of jumps per year)
    gamma (float): Mean of the jump size (log-normal jump)
    delta (float): Standard deviation of the jump size
    n (int): Number of time steps

    Returns:
    np.ndarray: Simulated stock prices

    """
    # Initialize arrays to store the simulated path
    S = np.zeros(n)
    S[0] = S0
    
    # Simulate Brownian motion for the continuous part
    dW = np.random.normal(0, np.sqrt(dt), n-1)
    
    # Simulate Poisson process for the jump part
    dN = np.random.poisson(lam * dt, n-1)
    
    # Simulate jump sizes (log-normal distribution for jumps)
    J = np.exp(np.random.normal(gamma, delta, n-1))
    
    for i in range(1, n):
        # Continuous part (Brownian motion)
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[i-1])
        
        # Jump part (if a jump occurs, dN[i-1] will be 1)
        if dN[i-1] > 0:
            S[i] *= J[i-1]  # Apply jump (multiply by the jump size)
        
    return S


def mjd_path(N_prime, C, t, dt, mjd_par, seed_path):
    '''
    It simulates the entire path of a MJD with regimes switch.
    
    '''
    np.random.seed(seed_path)
    # array of prices
    s = np.zeros(N_prime + 1)
    # initial stock price
    s[0] = 1
    s_0 = s[0]
    start_index = 0
    stop_index = 1

    for k in range(1, N_prime+1):
        if k == N_prime:
            s[start_index : stop_index + 1] = mjd(s_0, mjd_par[C[k]][0], mjd_par[C[k]][1], mjd_par[C[k]][2], mjd_par[C[k]][3], mjd_par[C[k]][4], len(t[start_index : stop_index + 1]), dt)

        elif C[k] == C[k+1]:
            stop_index = k+1

        else:
            s[start_index : stop_index + 1] = mjd(s_0, mjd_par[C[k]][0], mjd_par[C[k]][1], mjd_par[C[k]][2], mjd_par[C[k]][3], mjd_par[C[k]][4], len(t[start_index : stop_index + 1]), dt)
            #updates
            start_index = k
            s_0 = s[k]
            stop_index = k + 1
            
    return s



def synthetic_path_generation(path, h1, h2):
    # for both W k-means and M k-means

    N_prime, M = data_par(N, h1, h2)
    t = timestep[: N_prime + 1]

    subsequences, theo_labels, labels_prices = generate_regimes(N_prime, subseq_length = l_regime)

    if path == 'GBM':
        # to ensure reproducibility
        path_seed = ask_path_seed()
        print(f"Seed path scelto: {path_seed}")  
        prices = gbm_path(N_prime, labels_prices, t, dt, gbm_par, path_seed) 
        log_returns = np.diff(np.log(prices))

        # plot price path with regime switch
        directory_path = f'figures/{path}/path_seed_{path_seed}'
        ensure_directory_exists(directory_path)
        synthetic_price_path_plot(t, prices, subsequences, directory_path)
        synthetic_log_returns_plot(t, log_returns, subsequences, directory_path)


        gbm_information = {'prices': prices, 'log_returns': log_returns, 't': t,
        'subsequences' : subsequences, 'theo_labels': theo_labels, 'labels_prices': labels_prices, 'path_seed': path_seed}
        return M, gbm_information

    elif path == 'MJD':
        # to ensure reproducibility
        path_seed = ask_path_seed()
        print(f"path seed scelto: {path_seed}")  
        prices = mjd_path(N_prime, labels_prices, t, dt, mjd_par, path_seed) 
        log_returns = np.diff(np.log(prices))

        # plot price path with regime switch
        directory_path = f'figures/{path}/path_seed_{path_seed}'
        ensure_directory_exists(directory_path)
        synthetic_price_path_plot(t, prices, subsequences, directory_path)
        synthetic_log_returns_plot(t, log_returns, subsequences, directory_path)

        mjd_information = {'prices': prices, 'log_returns': log_returns, 't': t,
        'subsequences' : subsequences, 'theo_labels': theo_labels, 'labels_prices': labels_prices, 'path_seed': path_seed}
        return M, mjd_information
    


def real_path_processing(path, h1, h2):
    # for both M k-means and W k-means
    # import real data
    df = pd.read_csv('real_data/' + path.lower() + '_time_series.txt')
    s = df['price'].values
    timestep_real = df['time'].values

    N_real = len(timestep_real)
    # dt_real = timestep_real[1] - timestep_real[0]
    N_prime, m = data_par(N_real, h1, h2)

    # data for the analysis
    t = timestep_real[: N_prime + 1]
    s = s[: N_prime + 1]
    log_returns = np.diff(np.log(s))

    # plots
    directory_path = f'figures/{path}'
    ensure_directory_exists(directory_path)
    real_price_path_plot(t, s, directory_path)
    real_log_returns_plot(t, log_returns, directory_path)

    path_information = {'prices': s, 'log_returns': log_returns, 't': t}
    return m, path_information