import numpy as np

def set_proba(feature_list, main_branches = None):
    # Define probas of picking for each element of a feature.
    if main_branches == 'keep' or main_branches == 'remove':
        feature_list = feature_list.apply(lambda x: np.array(x)[np.all(np.array(x) >= 0.001, axis=-1)])    

    bar_lengths = feature_list.apply(lambda x: np.abs(np.array(x)[:, 1] - np.array(x)[:, 0]) if x is not None else None)
    probas = bar_lengths.apply(lambda x: x/sum(np.sort(x, axis=-1)))
    return probas

def subsample_w_replacement(feature_list, probas, k_elements, n_samples, rand_seed = 0, main_branches = None):
    # Subsample each feature n_samples time, following probas, to create a list of subfeatures with k_elements.
    np.random.seed(rand_seed)

    subsampled_features = []
    for feature, proba in zip(feature_list, probas):
        feature = np.array(feature)
        if k_elements < 1:
            k_elements = int(k_elements*feature.shape[0])
        if main_branches == 'keep':
            main_branches_mask = np.any(feature < 0.001, axis=-1)
            main_branches_indices = np.where(main_branches_mask)[0]
        else:
            main_branches_indices = np.array([], dtype=int)
        indices_list = [np.hstack(([np.random.choice(len(proba), p=proba) for _ in range(k_elements)], main_branches_indices)) for _ in range(n_samples)] 
        subsamples = []
        for indices in indices_list:
            subsamples.append(feature[indices])
        subsampled_features.append(subsamples)
    return subsampled_features