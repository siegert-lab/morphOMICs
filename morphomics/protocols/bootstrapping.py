"""
morphomics : bootstrapping tools

Author: Ryan Cubero
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from math import comb
from itertools import combinations

bootstrap_methods = {
    "mean": lambda arr: np.mean(arr),
    "median": lambda arr: np.median(arr),
    "max": lambda arr: np.amax(arr),
    "min": lambda arr: np.amin(arr),
    "mean_axis": lambda arr, ax: np.mean(arr, axis=ax),
    "median_axis": lambda arr, ax: np.median(arr, axis=ax),
    "max_axis": lambda arr, ax: np.amax(arr, axis=ax),
    "min_axis": lambda arr, ax: np.amin(arr, axis=ax),
}

def _bootstrap_feature(bootstrap_bag, feature_type, bootstrap_method = bootstrap_methods["mean_axis"]):
    '''The function `_bootstrap_feature` collects bars or averages over features based on the specified
    data type.
    
    Parameters
    ----------
    _b_frame
        The `_b_frame` parameter is likely a DataFrame or a similar data structure that contains the data
    for analysis. It could be used to store information about the features being analyzed, such as
    measurements or characteristics of a sample.
    morphology_list
        The `morphology_list` parameter likely contains a list of morphology features or characteristics
    that are used in the bootstrapping process. These features could include measurements such as
    length, width, area, volume, etc., depending on the context of the code and the specific application
    it is being used for.
    _feature
        The `_feature` parameter is a feature that is being used in the `_bootstrap_feature` function. It
    is a specific attribute or characteristic that is being analyzed or processed within the function.
    _dtype
        The `_dtype` parameter is used to specify the type of data being processed. In this code snippet,
    it is checked to determine whether the data type is "bars" or not, and different functions are
    called based on this condition.
    
    Returns
    -------
        The function `_bootstrap_feature` returns the `bootstrapped_feature` variable, which is either the
    result of `_collect_bars` function if the `_dtype` is "bars", or the result of
    `_average_over_features` function if `_dtype` is not "bars".
    
    '''
    if feature_type == "bars":
        bootstrapped_feature = bootstrap_bag
    elif feature_type == "array":
        bootstrapped_feature = bootstrap_method(bootstrap_bag, 0)
    else:
        bootstrapped_feature = bootstrap_method(bootstrap_bag)

    return bootstrapped_feature



def get_bootstrap_frame(
    info_frame,
    feature_to_bootstrap,
    bootstrap_conditions,
    N_bags,
    replacement,
    n_samples,
    rand_seed=0,
    ratio=None,
):

    np.random.seed(rand_seed)

    _feature, _dtype = feature_to_bootstrap
    assert (
        _feature in info_frame.keys()
    ), "There is no `%s` column in info_frame. Make sure that you have loaded the data properly."%_feature
    assert _dtype in ['bars', 'scalar', 'array'], "%s is not a valid dtype for %s"%(_dtype, _feature)
    
    # If bootstrap_conditions is empty, bootstrap on all the conditions in bootstrap_conditions.
    if len(bootstrap_conditions) == 0:
        bootstrap_conditions = ['Region', 'Model', 'Sex']
    # Create a dataframe with the necessary data for bootstrap.
    keys_list = [*bootstrap_conditions, feature_to_bootstrap[0]]
    _info_frame = info_frame.copy()
    _info_frame = _info_frame[keys_list]
    
    # Creating condition for bootstrap.
    _info_frame['condition'] = _info_frame[bootstrap_conditions].apply(lambda x: '-'.join(x), axis=1)
    condition_list = _info_frame['condition'].unique()
    
    bootstrap_frame_list = []
    for condition in condition_list:
        print("Performing bootstrapping for %s..." % condition)
        # Get the lis of the indxs of the samples from the condition
        pop_idxs = _info_frame.loc[_info_frame["condition"] == condition].index

        pop_length = len(pop_idxs)
        print("There are %d morphologies to bootstrap..." % pop_length)

        if ratio > 0:
            n_samples = int(ratio * pop_length)
        
        # Get the list of the bags. A bag is composed of randomly chose sampled indxs.    
        # But if the nb of samples is higher than the size of the pop, n_samples is reajusted.
        if not replacement and n_samples > pop_length:
            sampled_idxs_list = [np.random.choice(pop_idxs, pop_length, replace=replacement) 
                                for _ in range(N_bags)]
        
        sampled_idxs_list = [np.random.choice(pop_idxs, n_samples, replace=replacement) 
                             for _ in range(N_bags)]

        bootstraped_bag_list = []
        # Iterate through the N_bags.
        for sampled_idxs in sampled_idxs_list:
            bootstrap_bag = _info_frame.loc[sampled_idxs][feature_to_bootstrap[0]]
            bootstrap_bag =  np.vstack(bootstrap_bag)
            bootstraped_bag = _bootstrap_feature(bootstrap_bag, feature_to_bootstrap[1])
            bootstraped_bag_list.append(bootstraped_bag)
        
        condition_bootstrap_frame = pd.DataFrame(
            columns = bootstrap_conditions + ['condition', 'bootstrap_indices', feature_to_bootstrap[0]]
        )
        condition_bootstrap_frame['condition'] = N_bags * [condition]
        condition_bootstrap_frame['bootstrap_indices'] = sampled_idxs_list
        condition_bootstrap_frame[feature_to_bootstrap[0]] = bootstraped_bag_list
        # Get the list of the conditions.
        conditions = condition.split('-')
        for cond_key, cond in zip(bootstrap_conditions, conditions):
            condition_bootstrap_frame[cond_key] = cond
    
        bootstrap_frame_list.append(condition_bootstrap_frame)

    bootstrap_frame = pd.concat(bootstrap_frame_list, axis=0, ignore_index=True)

    return bootstrap_frame
