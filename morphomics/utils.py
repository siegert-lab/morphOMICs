"""
Contains all the commonly used functions and data
useful for multiple tmd modules.
"""
import numpy as np

term_dict = {"x": 0, "y": 1, "z": 2}

tree_type = {1: "soma", 2: "axon", 3: "basal", 4: "apical"}

array_operators = {
    # Make sure that arr is a numpy array
    "<": lambda arr, a: np.array(arr) < a,
    "<=": lambda arr, a: np.array(arr) <= a,
    "==": lambda arr, a: np.array(arr) == a,
    ">=": lambda arr, a: np.array(arr) >= a,
    ">": lambda arr, a: np.array(arr) > a,
}

distances = {
    "l1": lambda t1, t2: np.linalg.norm(np.subtract(t1, t2), 1),
    "l2": lambda t1, t2: np.linalg.norm(np.subtract(t1, t2), 2),
}

scipy_metric = {
    "l1": "cityblock",
    "l2": "euclidean",
}

barcode_dist = {"bd": "bottleneck_distance"}

norm_methods = {
    # Returns the normalization factor based on the norm_method
    "max": lambda arr: np.amax(arr),
    "sum": lambda arr: np.sum(arr),
    "min": lambda arr: np.amin(arr),
    "ave": lambda arr: np.mean(arr),
    "std": lambda arr: np.std(arr),
    "l1": lambda arr: np.sum(np.abs(arr)),
    "l2": lambda arr: np.sqrt(np.dot(arr, arr)),
    "id": lambda arr: np.ones_like(arr),
}

vectorization_codenames = {
    # Returns a codename for a vectorization method name 
    "persistence_image" : "pi",
    "betti_curve" : "bc",
    "life_entropy_curve" : "lec",
    "lifespan_curve" : "lsc",
    "stable_ranks" : "sr",
    "betti_hist" : "bh",
    "lifespan_hist": "lh"

}
    
    
def inquire_numbers_per_layer(_info_frame):
    needed_columns = ["Model", "Time", "Region"]
    for _cols in needed_columns:
        assert _cols in _info_frame.columns, '%s not in _info_frame'%_cols
    
    for model in _info_frame.Model.unique():
        _frame = _info_frame.loc[_info_frame.Model == model]
        for timepoints in _frame.Time.unique():
            print(
                model,
                timepoints,
            )
            print(
                _frame.loc[_frame.Time == timepoints].Region.value_counts(),
            )
