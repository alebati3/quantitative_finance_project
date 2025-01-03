import numpy as np

# optimization of the previous code by using ChatGPT
def opt_counter(kmeans, n, M, h_1, h_2, o=True):


    # Define the time indices for the sliding window
    time_indices = np.arange(n)[:, None] - (h_1 - h_2) * np.arange(M)[None, :]

    # Mask invalid indices
    valid_mask = (time_indices >= 0) & (time_indices <= h_1)

    # Use the valid_mask to filter time indices
    filtered_time_indices = time_indices * valid_mask

    # Create the labels array, repeated across all k for efficient processing
    labels_repeated = np.tile(kmeans.labels_, (n, 1))

    # Use the valid mask to apply the labels where indices are valid
    filtered_labels = np.where(valid_mask, labels_repeated, -1)

    # Count occurrences of each label
    r_counter_0 = np.sum(filtered_labels == 0, axis=1)
    r_counter_1 = np.sum(filtered_labels == 1, axis=1)

    # Combine the counts into a single array
    r_counter = np.stack((r_counter_0, r_counter_1), axis=1)
    
    if o:
    
        # Initialize s_counter with the same shape as r_counter
        s_counter = np.zeros((n+1, 2))

        # Handle the first element
        s_counter[0] = r_counter[0]

        # Handle the last element
        s_counter[-1] = r_counter[-1]

        # For all other elements, sum the current and previous elements
        s_counter[1:-1] = r_counter[:-1] + r_counter[1:]


        return r_counter, s_counter
    
    else:
        
        return r_counter

def compute_accuracy_scores(r_counter, off_regime_index, on_regime_index, theo_labels):
    # regime-off accuracy score (ROFS)
    ROFS = np.sum(r_counter[theo_labels == 0].T[off_regime_index])/np.sum(r_counter[theo_labels == 0])
    

    # regime-off accuracy score (ROFS)
    RONS = np.sum(r_counter[theo_labels == 1].T[on_regime_index])/np.sum(r_counter[theo_labels == 1])
    

    # total accuracy (TA)
    TA = (np.sum(r_counter[theo_labels == 0].T[off_regime_index]) + np.sum(r_counter[theo_labels == 1].T[on_regime_index]))/np.sum(r_counter)
    
    return ROFS, RONS, TA

def compute_accuracy_scores_hmm(hmm_labels, theo_labels, off_regime_index, on_regime_index):
    ROFS = np.sum(hmm_labels[theo_labels == 0] == off_regime_index) / len(hmm_labels[theo_labels == 0])
    RONS = np.sum(hmm_labels[theo_labels == 1] == on_regime_index) / len(hmm_labels[theo_labels == 1])
    TA = (np.sum(hmm_labels[theo_labels == 0] == off_regime_index) + np.sum(hmm_labels[theo_labels == 1] == on_regime_index)) / len(hmm_labels)
    return ROFS, RONS, TA