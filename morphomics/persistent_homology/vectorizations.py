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
    ph, method="kde", std_isotropic=0.1, xlim=None, ylim=None, bw_method=None, weights=None, resolution=100
):
    """Create array of the persistence image.

    Args:
        ph: persistence diagram.
        method: whether to aggregate the diagram using Gaussian with covariance estimated from data (in kde fashion) or with isotropic gaussian
        std_isotropic: standard deviation of the isotropic gaussian.
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
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.transpose(ph)

    if method == "kde":
        kernel = stats.gaussian_kde(values, bw_method=bw_method, weights=weights)
        Z = np.reshape(kernel(positions).T, X.shape)
    elif method == "isotropic":
        Z = _pi_isotropic(ph, positions.T, std_isotropic)
        Z = np.reshape(Z, X.shape)

    return Z

def _pi_isotropic(ph, eval_points, std):
    """
    Compute KDE for a dataset of 2D points using an isotropic Gaussian kernel.

    Parameters:
    - ph: np.ndarray, shape (N, 2)
        Array of 2D points from the dataset.
    - eval_points: np.ndarray, shape (M, 2)
        Array of 2D points where the KDE should be evaluated.
    - std: float
        Standard deviation of the isotropic Gaussian kernel.

    Returns:
    - kde_values: np.ndarray, shape (M,)
        The KDE values at each of the evaluation points.
    """
    # Ensure input is numpy arrays
    data_points = np.asarray(ph)
    eval_points = np.asarray(eval_points)

    # Gaussian kernel normalization constant in 2D
    normalization = 1 / (2 * np.pi * std**2)

    # Compute pairwise squared distances using broadcasting
    diff = eval_points[:, np.newaxis, :] - data_points[np.newaxis, :, :]  # Shape: (M, N, 2)
    squared_distances = np.sum(diff**2, axis=2)  # Shape: (M, N)

    # Apply Gaussian kernel
    weights = np.exp(-squared_distances / (2 * std**2))  # Shape: (M, N)

    # Sum over the data points (axis=1) and normalize
    kde_values = normalization * np.sum(weights, axis=1)  # Shape: (M,)

    return kde_values


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
    ph = np.array(ph)
    ph[:, [0, 1]] = np.sort(ph[:, [0, 1]], axis=1)
    betti_c = np.array( [np.count_nonzero((ph[:, 0] <= x) & (ph[:, 1] >= x)) for x in t_list] )
    return betti_c, t_list

def lifespan_curve(ph, t_list = None, resolution = 1000):
    # The vectorization called lifespan curve.
    # Returns the lifespan curve of a barcode and the sub intervals on which it was computed.
    if t_list is None:
        t_list = np.linspace(np.min(ph), np.max(ph), resolution)
    bars_length = get_lengths(ph, type="abs")
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

def stable_ranks(bar_lengths, prob, maxL, disc_steps):
    """Compute the stable ranks of a barcode."""
    if len(bar_lengths) == 0:
        return np.zeros(disc_steps)
    
    if prob=="long":
        prob_bars = bar_lengths / bar_lengths.sum()
    elif prob=="short":
        prob_bars = (bar_lengths.max() - bar_lengths) / (bar_lengths.max() - bar_lengths).sum()
        
    x_values = np.linspace(0, maxL, num=disc_steps)

    if prob=="long" or prob=="short":
        sr = np.array( [bar_lengths.shape[0]*prob_bars[bar_lengths >= x].sum() for x in x_values] )
    else:
        sr = np.array( [np.count_nonzero(bar_lengths >= x) for x in x_values] )

    return sr

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

    bars_length = get_lengths(ph, type="abs")
    lifespan_h = [np.sum([
                        bar_len if m else 0.
                        for m, bar_len in zip(mask, bars_length)
                        ])
                    for mask in masks
                ]
    return lifespan_h, bins