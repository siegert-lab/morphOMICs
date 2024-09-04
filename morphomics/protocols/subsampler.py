import numpy as np
from morphomics.cells.tree.tree import Tree

## Barcodes

def set_proba(feature_list, main_branches = None):
    # Define probas of picking for each element of a feature.
    if main_branches == 'keep' or main_branches == 'remove':
        feature_list = feature_list.apply(lambda ph: np.array([bar if np.all(bar >= 0.001) else [0, 0] for bar in ph]))
    
    bar_lengths = feature_list.apply(lambda ph: np.abs(ph[:, 1] - ph[:, 0]) if ph is not np.nan else np.nan)
    probas = bar_lengths.apply(lambda ph: ph/sum(np.sort(ph, axis=-1)))
    return probas

def subsample_w_replacement(feature_list, probas, k_elements, n_samples, 
                            rand_seed = 51, main_branches = None):
    # Subsample each feature n_samples time, following probas, to create a list of subfeatures with k_elements.
    np.random.seed(rand_seed)

    subsampled_features = []
    for feature, proba in zip(feature_list, probas):
        feature = np.array(feature)

        if not isinstance(k_elements, int) :
            k = int(k_elements*feature.shape[0])
        else:
            k = k_elements

        if main_branches == 'keep':
            main_branches_mask = np.any(feature < 0.001, axis=-1)
            main_branches_indices = np.where(main_branches_mask)[0]
        else:
            main_branches_indices = np.array([], dtype=int)

        indices_list = [np.hstack(([np.random.choice(len(proba), p=proba) for _ in range(k)], main_branches_indices)) 
                        for _ in range(n_samples)] 
        subsamples = []
        for indices in indices_list:
            subsamples.append(feature[indices])
        subsampled_features.append(subsamples)
    return subsampled_features

## Trees

def subsample_trees(feature_list, type, number, n_samples, rand_seed = 51):
    np.random.seed(rand_seed)
    if type[0] == 'cut':
        n_samples = 1
    subsampled_features = feature_list.apply(lambda cell : [cell.neurites[0].subsample(type, number) for _ in range(n_samples)])
    return subsampled_features