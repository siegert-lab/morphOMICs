import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist, pdist, squareform
from morphomics import utils
from itertools import chain


def get_barcode_dist_matrix(barcodes, metric):
    ''' Compute the distance matrix of the barcodes with respect to a norm (e.g. wasserstein distance).

    Parameters
    ----------
    barcodes (list of np.array): the list of barcodes.
    metric (str): a key in dictionnary utils.barcode_dist.

    Returns
    -------
    dist_matrix (np.array): the paired distances of the barcodes based on the metric
    '''

    return

def get_vect_dist_matrix(vectors, metric = "l2"):
    ''' Compute the distance matrix of the vectors with respect to a norm (e.g. l2).

    Parameters
    ----------
    vectors (np.array): the array of row vectors.
    metric (str): a key in dictionnary utils.scipy_metric.

    Returns
    -------
    paired_dist (np.array): the paired distances of the vectors based on the metric
    '''
    paired_dist = pdist(vectors,
                        metric= utils.scipy_metric[metric])
    #norm(np.abs(np.subtract(results1, results2)), normalized)
    return paired_dist

def get_dist_mat(paired_dist):
    dist_matrix = squareform(paired_dist)   # Convert to a square form matrix
    return dist_matrix

def distance_stepped(ph1, ph2, order=1):
    """Calculate step distance difference between two ph."""
    bins1 = np.unique(list(chain(*ph1)))
    bins2 = np.unique(list(chain(*ph2)))
    bins = np.unique(np.append(bins1, bins2))
    results1 = np.zeros(len(bins) - 1)
    results2 = np.zeros(len(bins) - 1)

    for bar in ph1:
        for it, _ in enumerate(bins[:-1]):
            if min(bar) <= bins[it + 1] and max(bar) > bins[it]:
                results1[it] = results1[it] + 1

    for bar in ph2:
        for it, _ in enumerate(bins[:-1]):
            if min(bar) <= bins[it + 1] and max(bar) > bins[it]:
                results2[it] = results2[it] + 1

    return norm(np.abs(np.subtract(results1, results2)) * (bins[1:] + bins[:-1]) / 2, order)