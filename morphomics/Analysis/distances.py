import numpy as np
from scipy.spatial.distance import cdist
import utils

def get_vect_dist_matrix(vectors, metric):
    ''' Compute the distance matrix of the vectors with respect to a norm (e.g. l2).

    Parameters
    ----------
    vectors (np.array): the array of row vectors.
    metric (str): a key in dictionnary utils.scipy_metric.

    Returns
    -------
    dist_matrix (np.array): the distance matrix of the vectors based on the metric
    '''
    dist_matrix = cdist(vectors, vectors,
                        metric= utils.scipy_metric[metric])

    return dist_matrix

def get_barcode_dist_matrix(barcodes, metric):
    ''' Compute the distance matrix of the barcodes with respect to a norm (e.g. wasserstein distance).

    Parameters
    ----------
    barcodes (list of np.array): the list of barcodes.
    metric (str): a key in dictionnary utils.barcode_dist.

    Returns
    -------
    dist_matrix (np.array): the distance matrix of the barcodes based on the metric
    '''

    return