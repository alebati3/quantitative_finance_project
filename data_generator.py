import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from pre_processing_utils import *
from accuracy_utils import *
from plot_utils import *
from io_utils import *



# Funzione per generare un indice di partenza valido
def generate_start_index(A, subseq_length, used_indices):
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
    while True:
        # Genera un indice di partenza casuale
        start_index = np.random.randint(0, len(A) - subseq_length - 1)
        # Controlla se l'indice di partenza e l'indice finale (con buffer di 1) sono validi
        if all((start_index + i) not in used_indices for i in range(subseq_length + 1)):
            for i in range(subseq_length + 1):
                used_indices.add(start_index + i)
            return start_index

# Funzione principale
def generate_regimes(n, regime_length, num_subsequences=10, random_state=17):
    """
    Genera casualmente 10 intervalli di tempo distinti della stessa lunghezza.

    Parameters:
    - n: dimensione dell'array di input
    - subseq_length: lunghezza delle sottosequenze
    - num_subsequences: numero di sottosequenze da generare (default 10)

    Returns:
    - subsequences: lista di sottosequenze generate
    - B: etichette per i log-returns
    - C: etichette per i prezzi
    """
    subseq_length = regime_length + 1
    a = np.arange(0, n+1)
    np.random.seed(random_state)
    # Set per memorizzare gli indici di partenza usati
    used_indices = set()

    # Generazione delle sottosequenze random non sovrapposte con almeno un elemento di distanza
    subsequences = []
    for _ in range(num_subsequences):
        start_index = generate_start_index(a, subseq_length, used_indices)
        subsequences.append(a[start_index:start_index + subseq_length])

    subsequences = np.sort(np.array(subsequences), axis=0)
    
    # Label per i log-returns
    b = np.zeros(n)
    for sub in subsequences:
        b[sub[0]: sub[-1]] = 1    
    b = b.astype(int)

    # Label per i prezzi
    c = np.zeros(n+1)
    for sub in subsequences:
        c[sub] = 1    
    c = c.astype(int)

    return subsequences, b, c

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

def gbm_path(n, c, t, dt, gbm_par, seed_path=None):
    '''
    It simulates the entire path of a GBM with regimes switch.
    
    '''
    np.random.seed(seed_path)
    # array of prices
    s = np.zeros(n + 1)
    # initial stock price
    s[0] = 1
    s_0 = s[0]
    start_index = 0
    stop_index = 1

    for k in range(1, n+1):
        if k == n:
            s[start_index : stop_index + 1] = gbm(s_0, gbm_par[c[k]][0], gbm_par[c[k]][1], len(t[start_index : stop_index + 1]), dt)

        elif c[k] == c[k+1]:
            stop_index = k+1

        else:
            s[start_index : stop_index + 1] = gbm(s_0, gbm_par[c[k]][0], gbm_par[c[k]][1], len(t[start_index : stop_index + 1]), dt)
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


def mjd_path(n, C, t, dt, mjd_par, seed_path):
    '''
    It simulates the entire path of a MJD with regimes switch.
    
    '''
    np.random.seed(seed_path)
    # array of prices
    s = np.zeros(n + 1)
    # initial stock price
    s[0] = 1
    s_0 = s[0]
    start_index = 0
    stop_index = 1

    for k in range(1, n+1):
        if k == n:
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




def synthetic_path_generation(path):
    # for W k-means, M k-means and HMM

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

    regimes, theo_return_labels, theo_price_labels = generate_regimes(N, l_regime)

    # GBM parameters
    gbm_par = np.array(
        [[0.02, 0.2], #mu,sigma *bull-regime*
        [-0.02, 0.3]]) #mu,sigma *bear-regime*

    # MJD parameters
    mjd_par = np.array(
        [[0.05, 0.2, 5, 0.02, 0.0125], # (mu,sigma, lambda, gamma, delta) bull-regime
        [-0.05, 0.4, 10, -0.04, 0.1]]) # (mu,sigma, lambda, gamma, delta) bear-regime


    # to ensure reproducibility
    path_seed = ask_path_seed() 

    if path == 'GBM':

        prices = gbm_path(N, theo_price_labels, t, dt, gbm_par, path_seed) 
        log_returns = np.diff(np.log(prices))

        # plot price path with regime switch
        directory_path = f'figures/{path}/path_seed_{path_seed}'
        ensure_directory_exists(directory_path)

        synthetic_price_path_plot(t, prices, regimes, directory_path)
        synthetic_log_returns_plot(t, log_returns, regimes, directory_path)


        gbm_information = {'prices': prices, 'log_returns': log_returns, 't': t,
        'regimes' : regimes, 'theo_return_labels': theo_return_labels, 'path_seed': path_seed}
        return gbm_information

    elif path == 'MJD':

        prices = mjd_path(N, theo_price_labels, t, dt, mjd_par, path_seed) 
        log_returns = np.diff(np.log(prices))

        # plot price path with regime switch
        directory_path = f'figures/{path}/path_seed_{path_seed}'
        ensure_directory_exists(directory_path)
        synthetic_price_path_plot(t, prices, regimes, directory_path)
        synthetic_log_returns_plot(t, log_returns, regimes, directory_path)

        mjd_information = {'prices': prices, 'log_returns': log_returns, 't': t,
        'regimes' : regimes, 'theo_return_labels': theo_return_labels, 'path_seed': path_seed}
        return mjd_information
    


def import_real_data(path):
    # for  W k-means, M k-means and HMM
    # import real data
    df = pd.read_csv('real_data/' + path.lower() + '_time_series.txt')
    prices = df['price'].values
    t = df['time'].values
    log_returns = np.diff(np.log(prices))

    # plots
    directory_path = f'figures/{path}'
    ensure_directory_exists(directory_path)
    real_price_path_plot(t, prices, directory_path)
    real_log_returns_plot(t, log_returns, directory_path)

    path_information = {'prices': prices, 'log_returns': log_returns, 't': t}
    return path_information