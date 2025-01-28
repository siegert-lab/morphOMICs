import numpy as np
from scipy import stats
from morphomics.persistent_homology.ph_analysis import get_lengths
from morphomics.utils import array_operators as ops

def concatenate(ph_list):
    return np.vstack(ph_list)

def sort_ph(ph):
    """Sorts barcode according to decreasing length of bars."""
    return ph[np.argsort([bar[0] - bar[1] for bar in ph])].tolist()

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
            return np.array(barcode_length)
    
    return np.array([])

def tmd_scale(barcode, thickness):
    """Scale the first two components according to the thickness parameter.

    Only these components are scaled because they correspond to spatial coordinates.
    """
    scaling_factor = np.ones(len(barcode[0]), dtype=float)
    scaling_factor[:2] = thickness
    return np.multiply(barcode, scaling_factor).tolist()


def transform_ph_to_length(ph, keep_side="end"):
    """Transform a persistence diagram into a (end, length) equivalent diagram.

    If `keep_side == "start"`, return a (start_point, length) diagram.

    .. note::

        The direction of the diagram will be lost!
    """
    if keep_side == "start":
        # keeps the start point and the length of the bar
        return [[min(i), np.abs(i[1] - i[0])] for i in ph]
    else:
        # keeps the end point and the length of the bar
        return [[max(i), np.abs(i[1] - i[0])] for i in ph]


def transform_ph_from_length(ph, keep_side="end"):
    """Transform a persistence diagram into a (end_point, length) equivalent diagram.

    If `keep_side == "start"`, return a (start_point, length) diagram.

    .. note::

        The direction of the diagram will be lost!
    """
    if keep_side == "start":
        # keeps the start point and the length of the bar
        return [[i[0], i[1] - i[0]] for i in ph]
    else:
        # keeps the end point and the length of the bar
        return [[i[0] - i[1], i[0]] for i in ph]


def nosify(var, noise=0.1):
    r"""Adds noise to an instance of data.

    Can be used with a ph as follows:

    .. code-block:: Python

        noisy_pd = [add_noise(d, 1.0) if d[0] != 0.0
                    else [d[0],add_noise([d[1]],1.0)[0]] for d in pd]

    To output the new pd:

    .. code-block:: Python

        F = open(...)
        for d in noisy_pd:
            towrite = '%f, %f\n'%(d[0],d[1])
            F.write(towrite)
        F.close()
    """
    var_new = np.zeros(len(var))
    for i, v in enumerate(var):
        var_new[i] = stats.norm.rvs(v, noise)
    return var_new


