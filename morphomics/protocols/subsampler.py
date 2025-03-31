import numpy as np
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

def subsample_phs_w_replacement(ph_list, probas, k_elements, n_samples, 
                            rand_seed = None, main_branches = None):
    # Subsample each feature n_samples time, following probas, to create a list of subfeatures with k_elements.
    if rand_seed is None:
        np.random.seed(rand_seed)
    subsampled_features = []
    for ph, proba in zip(ph_list, probas):
        ph = np.array(ph)
        # Update the number of elements to subsample from morphology
        if not isinstance(k_elements, int) :
            k = max(1, int(k_elements * ph.shape[0]))
        else:
            k = k_elements

        # Keep or remove the main branches
        if main_branches == 'keep':
            main_branches_mask = np.any(ph < 0.001, axis=-1)
            main_branches_indices = np.where(main_branches_mask)[0]
        else:
            main_branches_indices = np.array([], dtype=int)

        # Sample the 
        indices_list = [np.hstack(([np.random.choice(len(proba), p=proba) for _ in range(k)] if sum(proba) > 0.9 else [], main_branches_indices)) 
                        for _ in range(n_samples)] 
        
        subsamples = [ph[indices.astype(int)] for indices in indices_list]
        subsampled_features.append(subsamples)
    return subsampled_features

## Trees

def subsample_trees(tree_list, _type, number, n_samples, rand_seed = None):
    #_type can be 'cut' or 'prune'
    # if cut then n_samples is 1 because it is deterministic
    # and number is:
        # if _type = prune then number is either the number of nodes to remove or the probability to remove a node
        # if _type = cut then the number is 
    if rand_seed is not None:
        np.random.seed(rand_seed)

    if _type == 'cut':
        n_samples = 1


    def subs(tree):
        sub_tree_list = []
        for _ in range(n_samples):
            tree_copy = tree.copy_tree()
            sub_tree = tree_copy.subsample_tree(_type, number)
            sub_tree_list.append(sub_tree)
        return sub_tree_list

    subsampled_trees = tree_list.apply(subs)

    return subsampled_trees


def subsample_cells(cell_list, _type, number, n_samples, rand_seed = None):
    #_type can be 'cut' or 'prune'
    # if cut then n_samples is 1 because it is deterministic
    # and number is:
        # if _type = prune then number is either the number of nodes to remove or the probability to remove a node
        # if _type = cut then the number is 
    if rand_seed is not None:
        np.random.seed(rand_seed)

    if _type == 'cut' or (_type == 'prune' and isinstance(number, int)):
        n_samples = 1
    
    def subs(cell):
        sub_neuron_list = []
        for _ in range(n_samples):
            cell_copy = cell.copy_neuron()
            cell_copy = cell_copy.combine_neurites()
            tree = cell_copy.neurites[0]
            sub_tree = tree.subsample_tree(_type, number)
            neu = Neuron()
            neu.append_tree(sub_tree)
            neu.set_soma(cell.soma)
            sub_neuron_list.append(neu)
        return sub_neuron_list

    subsampled_neurons = cell_list.apply(subs)

    return subsampled_neurons