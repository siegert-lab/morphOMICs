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
    if _dtype == "bars":
        bootstrapped_feature = _collect_bars(_b_frame[_feature], morphology_list)
    else:
        bootstrapped_feature = _average_over_features(np.array(_b_frame[_feature]), morphology_list, _dtype)
    return bootstrapped_feature
    

def _average_over_features(features, morphology_list, _dtype, method="mean"):
    '''The function `_average_over_features` calculates the average of selected features based on specified
    morphology list and data type.
    
    Parameters
    ----------
    features
        Features is a list of feature values, where each element in the list represents a feature.
    morphology_list
        The `morphology_list` parameter is a list of indices that specify which features to include in the
    calculation.
    _dtype
        The `_dtype` parameter is used to specify the type of data being processed. It can have two
    possible values: "scalar" or "array".
    method, optional
        The `method` parameter in the `_average_over_features` function is used to specify the method for
    collapsing the features. The default value for this parameter is "mean", but you can also pass other
    methods like "median", "sum", etc. depending on how you want to collapse the features.
    
    Returns
    -------
        The function `_average_over_features` returns the collapsed result of the input features based on
    the specified method and data type. The specific result returned depends on the conditions within
    the function, such as whether the data type is "scalar" or "array" and the method chosen for
    collapsing the features.
    
    '''
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
    '''The function `_collect_bars` takes a DataFrame `_b_frame` and a list of indices `morphology_list`,
    and returns a pooled list of bars based on the indices.
    
    Parameters
    ----------
    _b_frame
        The `_b_frame` parameter is likely a DataFrame containing data related to bars or bar morphology.
    The function `_collect_bars` takes this DataFrame as input along with a list of indices
    `morphology_list`. It then collects the bars corresponding to the indices in the `morphology_list`
    from the
    morphology_list
        The `morphology_list` parameter is a list of indices that are used to select specific rows from the
    `_b_frame` DataFrame. These selected rows are then concatenated together to form a list of pooled
    bars, which is returned by the `_collect_bars` function.
    
    Returns
    -------
        The function `_collect_bars` returns a list of pooled bars, which are collected from the `_b_frame`
    DataFrame based on the indices provided in the `morphology_list`.
    
    '''
    pooled_bars = []
    for _idx in morphology_list:
        pooled_bars += _b_frame.iloc[_idx]
    return pooled_bars


def _create_bootstrap_dataframe(_b_frame, _bs, morphology_idx, pooled_bars, bootstrap_resolution):
    '''The function `_create_bootstrap_dataframe` generates a DataFrame with bootstrapped indices and
    barcodes based on input parameters.
    
    Parameters
    ----------
    _b_frame
        The `_b_frame` parameter is likely a DataFrame containing data related to bootstrapping conditions.
    It seems to be used to extract specific values based on the bootstrap condition `_bs` and morphology
    index `morphology_idx`. The function `_create_bootstrap_dataframe` creates a new DataFrame
    `bootstrap_frame` with
    _bs
        _bs is a variable representing the bootstrap condition.
    morphology_idx
        Morphology_idx is a variable representing the index of morphology data.
    pooled_bars
        Pooled_bars likely refers to a collection of barcodes that have been combined or aggregated in some
    way. This could be a list, array, or dataframe containing the barcodes that have been pooled
    together for analysis.
    bootstrap_resolution
        The `bootstrap_resolution` parameter in the function `_create_bootstrap_dataframe` appears to be a
    list of column names that are used to create a DataFrame. It is used to define the columns in the
    DataFrame that will be created and populated with values from the input `_b_frame` DataFrame based
    on the conditions
    
    Returns
    -------
        a DataFrame named `bootstrap_frame` with columns for the bootstrap resolution, "Bootstrapped
    index", and "Barcodes". The function populates the "Bootstrapped index" and "Barcodes" columns with
    the values of `morphology_idx` and `pooled_bars` respectively. It then iterates over the bootstrap
    resolution values to populate the corresponding columns in the
    
    '''
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
    
    '''The function `get_subsampled_population_from_infoframe` performs bootstrapping or subsampling on a
    DataFrame based on specified conditions and features.
    
    Parameters
    ----------
    info_frame
        The `info_frame` parameter is a DataFrame containing information about the population you want to
    subsample. It likely includes columns such as morphologies, barcodes, and other relevant features
    for your analysis.
    feature_to_bootstrap
        The `feature_to_bootstrap` parameter specifies the feature from the `info_frame` that you want to
    bootstrap. It should be a tuple containing the name of the feature and its data type. The data type
    can be one of 'bars', 'scalar', or 'array'.
    condition_column
        The `condition_column` parameter in the `get_subsampled_population_from_infoframe` function refers
    to the column in the `info_frame` DataFrame that contains the conditions based on which you want to
    perform bootstrapping. This column is used to group the data for bootstrapping based on different
    bootstrap_conditions
        The `bootstrap_conditions` parameter in the `get_subsampled_population_from_infoframe` function is
    a list of conditions based on which the bootstrapping will be performed. If this list is empty, the
    function will bootstrap on all the conditions present in the `condition_column`. If specific
    conditions are
    bootstrap_resolution
        The `bootstrap_resolution` parameter in the `get_subsampled_population_from_infoframe` function is
    used to specify the resolution at which the bootstrapping should be performed. It is a list of
    column names in the `info_frame` that will be used to create unique bootstrap conditions for
    sampling.
    N_pop
        The `N_pop` parameter in the `get_subsampled_population_from_infoframe` function represents the
    size of the population to sample from during bootstrapping or subsampling. It determines the number
    of elements to include in each sample when resampling the data. This parameter is crucial for
    controlling the
    N_samples
        The `N_samples` parameter in the `get_subsampled_population_from_infoframe` function represents the
    number of subsamples to generate during the bootstrapping process. This parameter determines how
    many random samples will be drawn from the population for each bootstrap condition. The function
    will either perform random subsampling
    rand_seed, optional
        The `rand_seed` parameter in the `get_subsampled_population_from_infoframe` function is used to set
    the random seed for reproducibility of the random number generation. By setting a specific
    `rand_seed`, you can ensure that the random sampling done within the function will produce the same
    results
    ratio
        The `ratio` parameter in the `get_subsampled_population_from_infoframe` function is used to specify
    the ratio of the total number of morphologies to be used for bootstrapping. It is used to calculate
    the size of the population for bootstrapping based on this ratio. If `
    save_filename
        The `save_filename` parameter in the `get_subsampled_population_from_infoframe` function is used to
    specify the filename under which the bootstrapped morphologies will be saved as an output. If you
    provide a value for `save_filename`, the function will save the resulting `bootstrapped
    
    Returns
    -------
        The function `get_subsampled_population_from_infoframe` returns a DataFrame containing bootstrapped
    morphologies based on the input parameters and conditions specified in the function.
    
    '''
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
