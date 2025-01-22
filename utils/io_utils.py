import os
import pandas as pd


def ensure_directory_exists(directory_path):
    """
    Checks if a directory exists, and creates it if it does not.

    :param directory_path: Path of the directory to check/create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory {directory_path} already existed. You might have overwritten the data!")

def save_and_print_values(directory_path, ROFS, RONS, TA, dec=2, file_name="results.txt"):
    """
    Saves the provided values (ROFS, RONS, TA) to a text file and prints them to the terminal.
    Creates the directory and file if they do not exist.

    :param directory_path: Path to the directory where the file will be saved.
    :param ROFS: Value of ROFS to be saved.
    :param RONS: Value of RONS to be saved.
    :param TA: Value of TA to be saved.
    :param dec: Number of decimal places for rounding (default is 2).
    :param file_name: Name of the file to save the values (default is "results.txt").
    """
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)

    # Format the values
    content_file = (
        f"{file_path}\n"
        f"ROFS = {round(100 * ROFS, dec)} %\n"
        f"RONS = {round(100 * RONS, dec)} %\n"
        f"TA = {round(100 * TA, dec)} %\n"
    )

    content = (
        f"ROFS = {round(100 * ROFS, dec)} %\n"
        f"RONS = {round(100 * RONS, dec)} %\n"
        f"TA = {round(100 * TA, dec)} %\n"
    )

    # Print the values to the terminal
    print("\nResults:")
    print(content)

    # Write to the file
    with open(file_path, 'w') as file:
        file.write(content_file)
    
    print(f"Values have also been saved to {file_path}")

def save_validation_results(rons, rofs, ta, run_times, directory_path):
    # print the results as txt file
    df = pd.DataFrame({
        'RONS': 100*rons,
        'ROFS': 100*rofs,
        'TA': 100*ta,
        'RUNTIME': run_times
    })

    # Construct the full file path
    file_name = 'validation_results.txt'
    file_path = os.path.join(directory_path, file_name)
    df.to_csv(file_path, index=False)

def ask_h1_h2():
    # all algorithm
    """
    Chiede all'utente di inserire i valori per i parametri h_1 e h_2.
    Se l'utente lascia il campo vuoto, vengono utilizzati i valori di default.
    I parametri devono essere interi strettamente positivi con h_1 maggiore di h_2.

    Returns:
        tuple: Una tupla contenente i valori di h_1 e h_2.
    """
    while True:
        try:
            # Input e validazione di h_1
            h_1 = input("\nInserisci il valore di h_1 (default 35, deve essere un intero positivo): ").strip()
            h_1 = int(h_1) if h_1 else 35
            if h_1 <= 0:
                raise ValueError("h_1 deve essere un intero strettamente positivo.")

            print(f"Valore selezionato per h_1: {h_1}")

            while True:
                try:
                    # Input e validazione di h_2
                    h_2 = input(f"Inserisci il valore di h_2 (default 28, deve essere un intero positivo e minore di {h_1}): ").strip()
                    h_2 = int(h_2) if h_2 else 28
                    if h_2 <= 0:
                        raise ValueError("h_2 deve essere un intero strettamente positivo.")
                    if h_2 >= h_1:
                        raise ValueError(f"h_2 deve essere strettamente minore di h_1 ({h_1}).")

                    print(f"Valore selezionato per h_2: {h_2}")
                    return h_1, h_2
                except ValueError as e:
                    print(f"Errore: {e}. Riprova.")
        except ValueError as e:
            print(f"Errore: {e}. Riprova.")


def ask_path_seed():
    # for all the algorithm; synthetic data
    """
    Chiede all'utente di inserire il valore per il parametro seed_path.
    Se l'utente lascia il campo vuoto o scrive 'None', viene utilizzato il valore di default (None).
    In caso contrario, il valore deve essere un intero positivo.

    Returns:
        int or None: Il valore di seed_path oppure None se lasciato vuoto o specificato come 'None'.
    """
    while True:
        try:
            seed_path = input("\nInserisci il valore di seed_path (default None, oppure un intero positivo): ").strip()

            if not seed_path or seed_path.lower() == 'none':
                print("Valore selezionato per seed_path: None")
                return None

            seed_path = int(seed_path)
            if seed_path <= 0:
                raise ValueError("Il valore di seed_path deve essere un intero positivo.")

            print(f"Valore selezionato per seed_path: {seed_path}")
            return seed_path
        except ValueError as e:
            print(f"Errore: {e}. Riprova.")



def ask_w_param():
    """
    Prompts the user to enter values for the following parameters:
    - p: positive integer greater than zero (default 1)
    - clustering_seed: integer greater than or equal to zero or None (default None)
    - max_iter: positive integer greater than zero (default 400)
    - tol: float greater than zero but less than 1 (default 1e-6)

    Returns:
        dict: A dictionary containing the parameter values.
    """
    print('\nSelection of clustering parameters')
    while True:
        try:
            # Input and validation for p
            p = input("Enter the value of p (default 1, must be a positive integer greater than zero): ").strip()
            p = int(p) if p else 1
            if p <= 0:
                raise ValueError("p must be a positive integer greater than zero.")

            print(f"Selected value for p: {p}")

            # Input and validation for clustering_seed
            clustering_seed = input("Enter the value of clustering_seed (default None, integer >= 0 or None): ").strip()
            if not clustering_seed or clustering_seed.lower() == 'none':
                clustering_seed = None
                print("Selected value for clustering_seed: None")
            else:
                clustering_seed = int(clustering_seed)
                if clustering_seed < 0:
                    raise ValueError("clustering_seed must be an integer greater than or equal to zero or None.")
                print(f"Selected value for clustering_seed: {clustering_seed}")

            # Input and validation for max_iter
            max_iter = input("Enter the value of max_iter (default 400, must be a positive integer): ").strip()
            max_iter = int(max_iter) if max_iter else 400
            if max_iter <= 0:
                raise ValueError("max_iter must be a positive integer greater than zero.")

            print(f"Selected value for max_iter: {max_iter}")

            # Input and validation for tol
            tol = input("Enter the value of tol (default 1e-6, must be a float > 0 and < 1): ").strip()
            tol = float(tol) if tol else 1e-6
            if tol <= 0 or tol >= 1:
                raise ValueError("tol must be a float greater than zero but less than 1.")

            print(f"Selected value for tol: {tol}")

            return {
                "p": p,
                "clustering_seed": clustering_seed,
                "max_iter": max_iter,
                "tol": tol
            }
        except ValueError as e:
            print(f"Error: {e}. Please try again.")



def ask_m_param():
    """
    Prompts the user to enter values for the following parameters:
    - p: positive integer greater than 1 (default 4)
    - clustering_seed: integer greater than or equal to zero or None (default None)
    - max_iter: positive integer greater than zero (default 400)
    - tol: float greater than zero but less than 1 (default 1e-6)

    Returns:
        dict: A dictionary containing the parameter values.
    """
    print('\nSelection of clustering parameters')
    while True:
        try:
            # Input and validation for p
            p = input("Enter the value of p (default 4, must be a positive integer): ").strip()
            p = int(p) if p else 4
            if p <= 1:
                raise ValueError("p must be a positive integer greater than one.")

            print(f"Selected value for p: {p}")

            # Input and validation for clustering_seed
            clustering_seed = input("Enter the value of clustering_seed (default None, integer >= 0 or None): ").strip()
            if not clustering_seed or clustering_seed.lower() == 'none':
                clustering_seed = None
                print("Selected value for clustering_seed: None")
            else:
                clustering_seed = int(clustering_seed)
                if clustering_seed < 0:
                    raise ValueError("clustering_seed must be an integer greater than or equal to zero or None.")
                print(f"Selected value for clustering_seed: {clustering_seed}")

            # Input and validation for max_iter
            max_iter = input("Enter the value of max_iter (default 400, must be a positive integer): ").strip()
            max_iter = int(max_iter) if max_iter else 400
            if max_iter <= 0:
                raise ValueError("max_iter must be a positive integer greater than zero.")

            print(f"Selected value for max_iter: {max_iter}")

            # Input and validation for tol
            tol = input("Enter the value of tol (default 1e-6, must be a float > 0 and < 1): ").strip()
            tol = float(tol) if tol else 1e-6
            if tol <= 0 or tol >= 1:
                raise ValueError("tol must be a float greater than zero but less than 1.")

            print(f"Selected value for tol: {tol}")

            return {
                "p": p,
                "clustering_seed": clustering_seed,
                "max_iter": max_iter,
                "tol": tol
            }
        except ValueError as e:
            print(f"Error: {e}. Please try again.")



def ask_hmm_param():
    """
    Prompts the user to enter values for the following parameters:
    - clustering_seed: integer greater than or equal to zero or None (default None)
    - max_iter: positive integer greater than zero (default 100)
    - tol: float greater than zero but less than 1 (default 1e-2)

    Returns:
        dict: A dictionary containing the parameter values.
    """
    print('\nSelection of clustering parameters')
    while True:
        try:
            # Input and validation for clustering_seed
            clustering_seed = input("Enter the value of clustering_seed (default None, integer >= 0 or None): ").strip()
            if not clustering_seed or clustering_seed.lower() == 'none':
                clustering_seed = None
                print("Selected value for clustering_seed: None")
            else:
                clustering_seed = int(clustering_seed)
                if clustering_seed < 0:
                    raise ValueError("clustering_seed must be an integer greater than or equal to zero or None.")
                print(f"Selected value for clustering_seed: {clustering_seed}")

            # Input and validation for max_iter
            max_iter = input("Enter the value of max_iter (default 100, must be a positive integer): ").strip()
            max_iter = int(max_iter) if max_iter else 100
            if max_iter <= 0:
                raise ValueError("max_iter must be a positive integer greater than zero.")

            print(f"Selected value for max_iter: {max_iter}")

            # Input and validation for tol
            tol = input("Enter the value of tol (default 1e-2, must be a float > 0 and < 1): ").strip()
            tol = float(tol) if tol else 1e-2
            if tol <= 0 or tol >= 1:
                raise ValueError("tol must be a float greater than zero but less than 1.")

            print(f"Selected value for tol: {tol}")

            return {
                "clustering_seed": clustering_seed,
                "max_iter": max_iter,
                "tol": tol
            }
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
























