import numpy as np
from morphomics.cells.tree.tree import Tree
from morphomics.cells.neuron import Neuron

## Barcodes

def set_proba(feature_list, main_branches = None):
    # Define probas of picking for each element of a feature.
    if main_branches == 'keep' or main_branches == 'remove':
        feature_list = feature_list.apply(lambda ph: np.array([bar if np.all(bar >= 0.001) else [0, 0] for bar in ph]))
    
    bar_lengths = feature_list.apply(lambda ph: np.abs(ph[:, 1] - ph[:, 0]) if ph is not np.nan else np.nan)

    # The if is because some barcodes only have main branches (trunks) and thus the proba is 0
    probas = bar_lengths.apply(lambda ph: ph/sum(np.sort(ph, axis=-1)) if sum(np.sort(ph, axis=-1))>1e-5 else np.zeros((len(ph))))
    return probas

def subsample_w_replacement(feature_list, probas, k_elements, n_samples, 
                            rand_seed = None, main_branches = None):
    # Subsample each feature n_samples time, following probas, to create a list of subfeatures with k_elements.
    if rand_seed is None:
        np.random.seed(rand_seed)
    i = 0
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
        indices_list = [np.hstack(([np.random.choice(len(proba), p=proba) for _ in range(k)] if sum(proba) > 0.9 else [], main_branches_indices)) 
                        for _ in range(n_samples)] 
        subsamples = []
        for indices in indices_list:
            indices = indices.astype(int)
            subsamples.append(feature[indices])
        subsampled_features.append(subsamples)
    return subsampled_features

## Trees

def subsample_trees(feature_list, _type, number, n_samples, rand_seed = None):
    #_type can be 'cut' or 'prune'
    # if cut then n_samples is 1 because it is deterministic
    # and number is 
    # if prune then number is either the number of nodes or the probability to remove a node
    if rand_seed is None:
        np.random.seed(rand_seed)

    if _type[0] == 'cut':
        n_samples = 1

    def subs(cell):
        neuron_list = []
        for _ in range(n_samples):
            cell = cell.combine_neurites()
            tree = cell.neurites[0].subsample_tree(_type, number)
            neu = Neuron()
            neu.append_tree(tree, 3)
            neuron_list.append(neu)
        return neuron_list


    subsampled_features = feature_list.apply(subs)

    return subsampled_features