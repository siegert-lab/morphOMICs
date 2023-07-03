"""
morphomics : bootstrapping tools

Author: Ryan Cubero
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from math import comb
from itertools import combinations

from morphomics.Analysis.reduction import get_distance_array
from morphomics.utils import save_obj

bootstrap_methods = {
    "mean": lambda arr: np.mean(arr),
    "median": lambda arr: np.median(arr),
    "max": lambda arr: np.amax(arr),
    "min": lambda arr: np.amin(arr),
    "mean_axis": lambda arr, ax: np.mean(arr, axis=ax),
    "median_axis": lambda arr, ax: np.median(arr, axis=ax),
    "max_axis": lambda arr, ax: np.amax(arr, axis=ax),
    "max_axis": lambda arr, ax: np.amin(arr, axis=ax),
}

def _bootstrap_feature(_b_frame, morphology_list, _feature, _dtype):
    if _dtype == "bars":
        bootstrapped_feature = _collect_bars(_b_frame[_feature], morphology_list)
    else:
        bootstrapped_feature = _average_over_features(np.array(_b_frame[_feature]), morphology_list, _dtype)
    return bootstrapped_feature
    

def _average_over_features(features, morphology_list, _dtype, method="mean"):
    _features = [features[i] for i in morphology_list]
    _features = np.array(_features)

    if _dtype == "scalar":
        collapsed = bootstrap_methods[method](np.hstack(_features))
    elif _dtype == "array":
        collapsed = bootstrap_methods["%s_axis" % method](_features, 0)
    else:
        raise ValueError("Check the dimension of features...")

    return collapsed


def _collect_bars(_b_frame, morphology_list):
    pooled_bars = []
    for _idx in morphology_list:
        pooled_bars += _b_frame.iloc[_idx]
    return pooled_bars


def _create_bootstrap_dataframe(_b_frame, _bs, morphology_idx, pooled_bars, bootstrap_resolution):
    bootstrap_frame = pd.DataFrame(
        columns=bootstrap_resolution + ["Bootstrapped index", "Barcodes"]
    )
    bootstrap_frame["Bootstrapped index"] = morphology_idx
    bootstrap_frame["Barcodes"] = pooled_bars

    for _bc in bootstrap_resolution:
        bootstrap_frame[_bc] = _b_frame.loc[
            _b_frame["bootstrap_condition"] == _bs
        ][_bc].values[0]
        
    return bootstrap_frame


def get_subsampled_population_from_infoframe(
    info_frame,
    feature_to_bootstrap,
    condition_column,
    bootstrap_conditions,
    bootstrap_resolution,
    N_pop,
    N_samples,
    rand_seed=0,
    ratio=None,
    save_filename=None,
):
    """Generates bootstrapped barcodes from the info_frame based on bootstrap_conditions with bootstrap_resolution.

    Args:
        info_frame (DataFrame): _description_
        feature_to_bootstrap (list, str)
        condition_column (str): _description_
        bootstrap_conditions (list, str): _description_
        bootstrap_resolution (list, str): _description_
        N_pop (int): _description_
        N_samples (int): _description_
        rand_seed (int): _description_. Defaults to 0.
        ratio (float, optional): _description_. Defaults to None.
        save_filename (str, optional): _description_. Defaults to None.

    Returns:
        DataFrame: _description_
    """
    np.random.seed(rand_seed)

    _feature, _dtype = feature_to_bootstrap
    assert (
        _feature in info_frame.keys()
    ), "There is no `%s` column in info_frame. Make sure that you have loaded the data properly."%_feature
    assert _dtype in ['bars', 'scalar', 'array'], "%s is not a valid dtype for %s"%(_dtype, _feature)
    
    # if bootstrap_conditions is empty, bootstrap on all the conditions in condition_column
    if len(bootstrap_conditions) == 0:
        _b_frame = info_frame.copy()
    else:    
        _b_frame = pd.concat(
            [
                info_frame.loc[info_frame[condition_column] == _conds]
                for _conds in bootstrap_conditions
            ],
            ignore_index=True,
        )
    
    _b_frame = _b_frame.loc[_b_frame["Morphologies"].notna()].reset_index(drop=True)

    # create the conditions for bootstrapping
    _b_frame["bootstrap_condition"] = ""
    for _bc in bootstrap_resolution:
        _b_frame["bootstrap_condition"] += _b_frame[_bc]
        if _bc != bootstrap_resolution[-1]:
            _b_frame["bootstrap_condition"] += "_"
    bs_conds = _b_frame["bootstrap_condition"].unique()

    bootstrap_frame = {}

    for _bs in bs_conds:
        print("Performing bootstrapping for %s..." % _bs)
        morphology_idx = _b_frame.loc[_b_frame["bootstrap_condition"] == _bs].index

        N = len(morphology_idx)
        print("There are %d morphologies to bootstrap..." % N)

        if ratio is not None:
            N_pop = int(ratio * N)
            
            
        subsampled = []
        subsampled_index = []
        
        # if N_pop >= N, calculate the average persistence image
        if N_pop >= N:
            print(
                "The bootstrap size is greater than or equal to the original population size. Calculating the average persistence image..."
            )
            subsampled.append(_bootstrap_feature(_b_frame, morphology_idx, _feature, _dtype))
            # subsampled.append(_collect_bars(_b_frame, morphology_idx))
            subsampled_index.append(morphology_idx)
            
            bootstrap_frame[_bs] = _create_bootstrap_dataframe(_b_frame, _bs, subsampled_index, subsampled, bootstrap_resolution)
            print("...done! \n")
            continue
        
        
        max_possible = comb(N, N_pop)
        
        # if max_possible > N_samples: then bootstrap
        # else, enumerate all subsamples and subsample
        if max_possible >= N_samples:
            print(
                "Performing subsampling by random selection..."
            )
            for kk in np.arange(N_samples):
                # draw a subset of cells from the population and save the cell indices
                morphology_list = morphology_idx[
                    np.sort(np.random.choice(np.arange(N), N_pop, replace=False))
                ]
                subsampled_index.append(morphology_list)
                
                # perform the averaging over the subset of cells and save
                pooled_samples = _bootstrap_feature(_b_frame, morphology_list, _feature, _dtype)
                subsampled.append(pooled_samples)

        else:
            print(
                "Enumerating all possible combinations..."
            )
            for _idx in combinations(np.arange(N), N_pop):
                morphology_list = morphology_idx[list(_idx)]
                subsampled_index.append(morphology_list)
                
                # perform the averaging over the subset of cells and save
                pooled_samples = _bootstrap_feature(_b_frame, morphology_list, _feature, _dtype)
                subsampled.append(pooled_samples)

        bootstrap_frame[_bs] = _create_bootstrap_dataframe(_b_frame, _bs, subsampled_index, subsampled, bootstrap_resolution)
        print("...done! \n")
        
        
    bootstrapped_morphologies = pd.concat(
        [bootstrap_frame[_bs] for _bs in bs_conds], ignore_index=True
    )
    
    bootstrapped_morphologies = bootstrapped_morphologies.loc[
        bootstrapped_morphologies["Barcodes"].notna()
    ].reset_index(drop=True)

    if save_filename is not None:
        save_obj(bootstrapped_morphologies, "%s" % (save_filename))

    return bootstrapped_morphologies


# def _surprise(p):
#     if p == 0:
#         return 0
#     else:
#         return -p * np.log2(p)


# def _mixing_entropy(clustering, N_samples):
#     original_labels = np.array([1] * N_samples + [2] * N_samples)

#     p1_size = len(np.where(clustering == 1)[0])
#     p2_size = len(np.where(clustering == 2)[0])
#     N_ = p1_size + p2_size

#     p1 = np.mean(original_labels[np.where(clustering == 1)[0]] == 1)
#     p2 = np.mean(original_labels[np.where(clustering == 2)[0]] == 1)

#     cluster_entropy = (p1_size / (p1_size + p2_size)) * (
#         _surprise(p1) + _surprise(1.0 - p1)
#     ) + (p2_size / (p1_size + p2_size)) * (_surprise(p2) + _surprise(1.0 - p2))

#     return cluster_entropy


# def calculate_mixing_entropy(ph1, ph2, parameters, rand_seed=10):
#     N_pop = parameters["N_pop"]
#     N_samples = parameters["N_samples"]
#     if not parameters["linkage"]:
#         parameters["linkage"] = "single"

#     phs_cluster_1, _ = get_subsampled_population(ph1, N_pop, N_samples, rand_seed)
#     phs_cluster_2, _ = get_subsampled_population(ph2, N_pop, N_samples, rand_seed)

#     phs_batched = list(phs_cluster_1) + list(phs_cluster_2)

#     X = get_distance_array(phs_batched, xlims=[0, 200], ylims=[0, 200])
#     linked = linkage(X, parameters["linkage"], optimal_ordering=False)
#     clustering = fcluster(linked, t=2, criterion="maxclust")

#     cluster_entropy = _mixing_entropy(clustering)

#     return cluster_entropy
