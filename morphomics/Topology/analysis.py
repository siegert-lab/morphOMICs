"""
morphomics Topology analysis algorithms implementation
Adapted from https://github.com/BlueBrain/TMD
"""
# pylint: disable=invalid-slice-index
import copy
import math
from itertools import chain
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy import stats
from morphomics.utils import array_operators as ops
from morphomics.utils import norm_methods


def get_lengths(ph):
    """
    Returns the lengths of the bars from the diagram
    """
    return np.array([np.abs(i[0] - i[1]) for i in ph])


def collapse(ph_list):
    """
    Collapses a list of ph diagrams
    into a single instance for plotting.
    """
    return [list(pi) for p in ph_list for pi in p]


def sort_ph(ph):
    """
    Sorts barcode according to decreasing length of bars.
    """
    return np.array(ph)[np.argsort([p[0] - p[1] for p in ph])].tolist()


def filter_ph(ph, cutoff, method="<="):
    """
    Cuts off bars depending on their length
    ph:
    cutoff:
    methods: "<", "<=", "==", ">=", ">"
    """
    barcode_length = []
    if len(ph) >= 1:
        lengths = get_lengths(ph)
        cut_offs = np.where(ops[method](lengths, cutoff))[0]
        if len(cut_offs) >= 1:
            barcode_length = [ph[i] for i in cut_offs]

        return barcode_length

    else:
        raise "Barcode is empty"


def get_limits(phs_list, coll=True):
    """Returns the x-y coordinates limits (min, max)
    for a list of persistence diagrams
    """
    if coll:
        ph = collapse(phs_list)
    else:
        ph = copy.deepcopy(phs_list)
    xlims = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    ylims = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]
    return xlims, ylims


def get_persistence_image_data(
    ph,
    xlims=None,
    ylims=None,
    norm_factor=None,
    bw_method=None,
    norm_method="max",
    weights=None,
):
    """
    Create the data for the generation of the persistence image.
    ph: persistence diagram
    norm_factor: persistence image data are normalized according to this.
        If norm_factor is provided the data will be normalized based on this,
        otherwise they will be normalized to 1.
    xlims, ylims: the image limits on x-y axes.
        If xlims, ylims are provided the data will be scaled accordingly.
    bw_method: The method used to calculate the estimator bandwidth for the gaussian_kde.
    norm_method: The method used to normalize the persistence images (chosen between "max" or "sum")
    weights: Any weighting parameter that will be attached to each bar
    """
    if xlims is None or xlims is None:
        xlims, ylims = get_limits(ph, coll=False)

    X, Y = np.mgrid[xlims[0] : xlims[1] : 100j, ylims[0] : ylims[1] : 100j]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values, bw_method=bw_method, weights=weights)

    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    if norm_factor is None:
        norm_factor = norm_methods[norm_method](Z)

    return Z / norm_factor


def get_image_diff_data(Z1, Z2, normalized=True, norm_method="max"):
    """
    Takes as input two images as exported from the gaussian kernel
    plotting function, and returns their difference: diff(Z1 - Z2)
    """
    if normalized:
        Z1_norm = norm_methods[norm_method](Z1)
        Z2_norm = norm_methods[norm_method](Z2)

        Z1 = Z1 / Z1_norm
        Z2 = Z2 / Z2_norm
    return Z1 - Z2


def get_image_add_data(Z1, Z2, normalized=True, norm_method="max"):
    """
    Takes as input two images
    as exported from the gaussian kernel
    plotting function, and returns
    their sum: add(Z1 - Z2)
    """
    if normalized:
        Z1_norm = norm_methods[norm_method](Z1)
        Z2_norm = norm_methods[norm_method](Z2)

        Z1 = Z1 / Z1_norm
        Z2 = Z2 / Z2_norm
    return Z1 + Z2


def get_average_persistence_image(
    ph_list, xlims=None, ylims=None, norm_factor=None, weighted=False
):
    """
    Plots the gaussian kernel of a population of cells
    as an average of the ph diagrams that are given.
    """
    im_av = False
    k = 1
    if weighted:
        weights = [len(p) for p in ph_list]
        weights = np.array(weights, dtype=float) / np.max(weights)
    else:
        weights = [1 for _ in ph_list]

    for weight, ph in zip(weights, ph_list):
        if not isinstance(im_av, np.ndarray):
            try:
                im = get_persistence_image_data(
                    ph, norm_factor=norm_factor, xlims=xlims, ylims=ylims
                )
                if not np.isnan(np.sum(im)):
                    im_av = weight * im
            except BaseException:  # pylint: disable=broad-except
                pass
        else:
            try:
                im = get_persistence_image_data(
                    ph, norm_factor=norm_factor, xlims=xlims, ylims=ylims
                )
                if not np.isnan(np.sum(im)):
                    im_av = np.add(im_av, weight * im)
                    k = k + 1
            except BaseException:  # pylint: disable=broad-except
                pass
    return im_av / k
