"""TMD Topology analysis algorithms implementation."""

# Copyright (C) 2022  Blue Brain Project, EPFL
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# pylint: disable=invalid-slice-index
import math
from itertools import chain

import numpy as np
from numpy.linalg import norm
from scipy import stats
from scipy.spatial.distance import cdist

from morphomics.persistent_homology.ph_analysis import get_lengths
from morphomics.persistent_homology.vectorizations import betti_hist

def find_apical_point_distance(ph):
    """Finds the apical distance (measured from the soma) based on the variation of the barcode."""
    # Computation of number of components within the barcode
    # as the number of bars with at least max length / 2
    lengths = get_lengths(ph)
    num_components = len(np.where(np.array(lengths) >= max(lengths) / 2.0)[0])
    # Separate the barcode into sufficiently many bins
    n_bins, counts = betti_hist(ph, num_bins=3 * len(ph))
    # Compute derivatives
    der1 = counts[1:] - counts[:-1]  # first derivative
    der2 = der1[1:] - der1[:-1]  # second derivative
    # Find all points that take minimum value, defined as the number of components,
    # and have the first derivative zero == no variation
    inters = np.intersect1d(np.where(counts == num_components)[0], np.where(der1 == 0)[0])
    # Find all points that are also below a positive second derivative
    # The definition of how positive the second derivative should be is arbitrary,
    # but it is the only value that works nicely for cortical cells
    try:
        best_all = inters[np.where(inters <= np.max(np.where(der2 > len(n_bins) / 100)[0]))]
    except ValueError:
        return 0.0

    if len(best_all) == 0 or n_bins[np.max(best_all)] == 0:
        return np.inf
    return n_bins[np.max(best_all)]


def find_apical_point_distance_smoothed(ph, threshold=0.1):
    """Finds the apical distance (measured from the soma) based on the variation of the barcode.

    This algorithm always computes a distance, even if there is no obvious apical point.
    The threshold corresponds to percent of minimum derivative variation that is used to select the
    minima.
    """
    bin_centers, data = barcode_bin_centers(ph, num_bins=100)

    # Gaussian kernel to smooth distribution of bars
    kde = stats.gaussian_kde(data)
    minimas = []

    # Compute first derivative
    der1 = np.gradient(kde(bin_centers))
    # Compute second derivative
    der2 = np.gradient(der1)

    while len(minimas) == 0:
        # Compute minima of distribution
        minimas = np.where(abs(der1) < threshold * np.max(abs(der1)))[0]
        minimas = minimas[der2[minimas] > 0]
        threshold *= 2.0  # if threshold was too small, increase and retry
    return bin_centers[minimas[0]]


def _symmetric(p):
    """Returns the symmetric point of a PD point on the diagonal."""
    return [(p[0] + p[1]) / 2.0, (p[0] + p[1]) / 2]


def matching_munkress_modified(p1, p2, use_diag=True):
    """Find matching components and the corresponding distance between the two input diagrams."""
    import munkres  # pylint: disable=import-outside-toplevel

    if use_diag:
        p1_enh = p1 + [_symmetric(i) for i in p2]
        p2_enh = p2 + [_symmetric(i) for i in p1]
    else:
        p1_enh = p1
        p2_enh = p2

    D = cdist(p1_enh, p2_enh)
    m = munkres.Munkres()
    indices = m.compute(np.copy(D))
    ssum = np.sum([D[i][j] for (i, j) in indices])  # pylint: disable=unsubscriptable-object

    return indices, ssum
