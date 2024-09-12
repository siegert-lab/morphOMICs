"""Topology vectorization algorithms."""

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

import numpy as np
from scipy import stats
from itertools import chain

from morphomics.persistent_homology.ph_analysis import get_limits, get_lengths, get_total_length


### 2D vectorizations

def persistence_image(
    ph, xlim=None, ylim=None, bw_method=None, weights=None, resolution=100
):
    """Create array of the persistence image.

    Args:
        ph: persistence diagram.
        xlim: The image limits on x axis.
        ylim: The image limits on y axis.
        bw_method: The method used to calculate the estimator bandwidth for the gaussian_kde.
        weights: weights of the diagram points
        resolution: number of pixels in each dimension

    If xlim, ylim are provided the data will be scaled accordingly.
    """
    if xlim is None or ylim is None:
        xlim, ylim = get_limits(ph)
    res = complex(0, resolution)
    X, Y = np.mgrid[xlim[0] : xlim[1] : res, ylim[0] : ylim[1] : res]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values, bw_method=bw_method, weights=weights)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    return Z

### 1D curve vectorizations

def _index_bar(bar, t):
    """Computes if a bar is present at time t."""
    if min(bar) <= t <= max(bar):
        return 1
    else:
        return 0

def betti_curve(ph, t_list=None, resolution=1000):
    """Computes the betti curve of a persistence diagram.
    Corresponding to the number of bars at each distance t.
    """
    if t_list is None:
        t_list = np.linspace(np.min(ph), np.max(ph), resolution)
    betti_c = [np.sum([_index_bar(bar, t) for bar in ph]) for t in t_list]
    return betti_c, t_list

def lifespan_curve(ph, t_list = None, resolution = 1000):
    # The vectorization called lifespan curve.
    # Returns the lifespan curve of a barcode and the sub intervals on which it was computed.
    if t_list is None:
        t_list = np.linspace(np.min(ph), np.max(ph), resolution)
    bars_length = get_lengths(ph, abs = False)
    # bar_differences = bar_differences.ravel().astype(float)
    lifespan_c = [np.sum([
                        bar_len if _index_bar(bar, t) else 0.
                        for bar, bar_len in zip(ph, bars_length)
                        ])
                    for t in t_list
                ]
    return lifespan_c, t_list

def _bar_entropy(bar, lifetime):
    """Absolute difference of a bar divided by lifetime."""
    Zn = np.abs(bar[0] - bar[1]) / lifetime
    return Zn * np.log(Zn)

def life_entropy_curve(ph, t_list=None, resolution=1000):
    """The life entropy curve, computes life entropy at different t values."""
    lifetime = get_total_length(ph)
    # Compute the entropy of each bar
    entropy = [_bar_entropy(bar, lifetime) for bar in ph]
    if t_list is None:
        t_list = np.linspace(np.min(ph), np.max(ph), resolution)
    entropy_c = [
        -np.sum([_index_bar(ph_bar, t) * e for (e, ph_bar) in zip(entropy, ph)])
        for t in t_list
    ]
    return entropy_c, t_list

# 1D ordered vectorization

def stable_ranks(ph, type = 'standard'):
    bars_length = get_lengths(ph, abs = False)
    if type == 'standard':
        bars_length_filtered = -bars_length
    elif type == 'abs':
        bars_length_filtered = np.abs(bars_length)
    elif type == 'positiv':
        bars_length_filtered = np.abs(bars_length[bars_length < 0])

    bars_length_sorted = np.sort(bars_length_filtered)[::-1]
    return bars_length_sorted

def histogram_stepped(ph):
    """Calculate step distance of ph data."""
    bins = np.unique(list(chain(*ph)))
    results = np.zeros(len(bins) - 1)

    for bar in ph:
        for it, _ in enumerate(bins[:-1]):
            if min(bar) <= bins[it + 1] and max(bar) > bins[it]:
                results[it] = results[it] + 1

    return results, bins

# 1D histogram vectorizations

def _mask_bars(ph, bins):
    ph = np.sort(ph)
    starts = ph[:,0]
    ends = ph[:,1]
    masks = []
    for bin in bins:
        mask_a = starts < bin[1]
        mask_b = bin[0] < ends
        mask = np.logical_and(mask_a, mask_b)
        masks.append(mask)
    return masks

def _subintervals(xlims, num_bins = 1000):
    # Generate the linspace array
    linspace_array = np.linspace(xlims[0], xlims[1], num_bins+1)
    # Create subintervals
    subintervals = np.array([[linspace_array[i], linspace_array[i + 1]] for i in range(num_bins)])
    return subintervals

def betti_hist(ph, bins = None, num_bins = 1000):
    if bins is None:
        xlims = [np.min(ph), np.max(ph)]
        bins = _subintervals(xlims=xlims, num_bins=num_bins)
    masks = _mask_bars(ph, bins)
    betti_h = np.sum(masks, axis=-1)
    return betti_h, bins

def lifespan_hist(ph, bins = None, num_bins = 1000):
    if bins is None:
        xlims = [np.min(ph), np.max(ph)]
        bins = _subintervals(xlims=xlims, num_bins=num_bins)
    masks = _mask_bars(ph, bins)

    bars_length = get_lengths(ph, abs = False)
    lifespan_h = [np.sum([
                        bar_len if m else 0.
                        for m, bar_len in zip(mask, bars_length)
                        ])
                    for mask in masks
                ]
    return lifespan_h, bins