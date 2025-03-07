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

def extract_and_convert(input_string):
    try:
        # Find the position of the first "_"
        underscore_pos = input_string.find("_")
        if underscore_pos == -1:
            underscore_pos = len(input_string)
        
        # Extract the part between the 2nd character and the first "_"
        extracted_part = input_string[1:underscore_pos]
        
        # Convert the extracted part to an integer
        result = int(extracted_part)
        
        return result
    except ValueError as e:
        print(f"Error: {e}")
        return None
    

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
    numeric_condition,
    numeric_condition_std,
    N_bags,
    replacement,
    n_samples,
    rand_seed=None,
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
    if numeric_condition:
        keys_list.append(numeric_condition)
    _info_frame = info_frame.copy()
    _info_frame = _info_frame[keys_list]
    
    if numeric_condition:
        _info_frame["Time"] = _info_frame["Time"].apply(extract_and_convert)

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
            if n_samples == 0:
                n_samples = 1

        if numeric_condition:
            min_val, max_val = _info_frame.loc[pop_idxs][numeric_condition].min(), _info_frame.loc[pop_idxs][numeric_condition].max()
            bins = np.linspace(min_val, max_val, N_bags)
            bins = [int(b) for b in bins]
            sampled_idxs_list = []
            for i in range(N_bags):
                bag_value = bins[i]
                time_diffs = np.abs(_info_frame.loc[pop_idxs]["Time"] - bag_value)
                # Compute probabilities by evaluating under a Gaussian distribution
                probabilities = np.exp(-time_diffs ** 2 / (2 * numeric_condition_std ** 2))
                probabilities /= probabilities.sum()  # Normalize to create a probability distribution
                

                sampled_idxs = np.random.choice(pop_idxs, n_samples, replace=replacement, p=probabilities)
                sampled_idxs_list.append(sampled_idxs)

        else:
            # Get the list of the bags. A bag is composed of randomly chose sampled indcs.    
            # But if the nb of samples is higher than the size of the pop, n_samples is reajusted.
            if not replacement and n_samples > pop_length:
                _size = pop_length
            else:
                _size = n_samples
            
            sampled_idxs_list = [np.random.choice(pop_idxs, size = _size, replace=replacement) 
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
        if numeric_condition:
            condition_bootstrap_frame[numeric_condition] = bins

        # Get the list of the conditions.
        conditions = condition.split('-')
        for cond_key, cond in zip(bootstrap_conditions, conditions):
            condition_bootstrap_frame[cond_key] = cond
    
        bootstrap_frame_list.append(condition_bootstrap_frame)

    bootstrap_frame = pd.concat(bootstrap_frame_list, axis=0, ignore_index=True)

    return bootstrap_frame
