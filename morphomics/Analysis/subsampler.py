import numpy as np

def set_proba(feature_list):
    bar_lengths = feature_list.apply(lambda x: np.abs(np.array(x)[:, 1] - np.array(x)[:, 0]) if x is not None else None)
    probas = bar_lengths.apply(lambda x: x/sum(np.sort(x, axis=-1)))
    return probas

def subsample_w_replacement(feature_list, probas, k_elements, n_samples, rand_seed = 0):
    np.random.seed(rand_seed)

    subsampled_features = []
    for feature, proba in zip(feature_list, probas):
        indices_list = [[np.random.choice(len(proba), p=proba) for _ in range(k_elements)] for _ in range(n_samples)] 
        subsamples = []
        feature = np.array(feature)
        for indices in indices_list:
            subsamples.append(feature[indices])
        subsampled_features.append(subsamples)
    return subsampled_features