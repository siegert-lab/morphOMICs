import numpy as np

# Define a function to filter out the extremes based on cutoff value
def remove_extremes_absolute(df, col_name, min = None, max = None):
    if min is None:
        min = - np.inf
    if max is None:
        max = np.inf
        
    # Filter out values outside the [low_limit, high_limit] range
    filtered_df = df[(df[col_name] >= min) & (df[col_name] <= max)]
    extreme_df = (df[col_name] < min) | (df[col_name] > max)
    return filtered_df, extreme_df

# Define a function to filter out the extremes based on percentiles
def remove_extremes_relative(df, col_name, low_percentile=0.01, high_percentile=0.99):
    # Loop through each column
    low_limit = df[col_name].quantile(low_percentile)
    high_limit = df[col_name].quantile(high_percentile)
        
    # Filter out values outside the [low_limit, high_limit] range
    filtered_df, extreme_df = remove_extremes_absolute(df, col_name, low_limit, high_limit)
    return filtered_df, extreme_df

# Process Trees

# Process Barcodes
def remove_small_barcodes(morphoframe, barcode_size_cutoff):
    print("Removing morphologies with barcode size less than %.2f..."%barcode_size_cutoff)
    if "nb_bras" not in morphoframe.keys():
        morphoframe["nb_bars"] = morphoframe.barcodes.apply(lambda x: len(x))
    morphoframe = morphoframe.query(
        "nb_bars >= @barcode_size_cutoff"
        ).reset_index(drop=True)
    return morphoframe

