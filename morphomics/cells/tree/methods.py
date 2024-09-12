"""TMD Tree's methods."""

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

import copy
from collections import OrderedDict
from itertools import starmap

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from morphomics.utils import distances

def _rd_w(p1, p2, w=(1.0, 1.0, 1.0), normed=True):
    """Return weighted euclidean distance between p1 and p2."""
    if normed:
        w = np.array(w) / np.linalg.norm(w)
    return np.dot(w, (np.subtract(p1, p2)))


# Points features to be used for topological extraction
def get_point_radial_distances_time(self, point=None, dim="xyz", zero_time=0, time=1):
    """Tree method to get radial distances from a point.

    If point is None, the soma surface -defined by
    the initial point of the tree- will be used
    as a reference point.
    """
    if point is None:
        point = []
        for d in dim:
            point.append(getattr(self, d)[0])
    point.append(zero_time)

    radial_distances = np.zeros(self.size(), dtype=float)

    for i in range(self.size()):
        point_dest = []
        for d in dim:
            point_dest.append(getattr(self, d)[i])
        point_dest.append(time)

        radial_distances[i] = distances['l2'](point, point_dest)

    return radial_distances


def get_point_weighted_radial_distances(self, point=None, dim="xyz", w=(1, 1, 1), normed=False):
    """Tree method to get radial distances from a point.

    If point is None, the soma surface -defined by
    the initial point of the tree- will be used
    as a reference point.
    """
    if point is None:
        point = []
        for d in dim:
            point.append(getattr(self, d)[0])

    radial_distances = np.zeros(size(self), dtype=float)

    for i in range(size(self)):
        point_dest = []
        for d in dim:
            point_dest.append(getattr(self, d)[i])

        radial_distances[i] = _rd_w(point, point_dest, w, normed)

    return radial_distances


def get_trunk_length(self):
    """Tree method to get the trunk (first section length)."""
    ways, end = self.get_sections_only_points()
    first_section_id = np.where(ways == 0)
    first_section_start = ways[first_section_id]
    first_section_end = end[first_section_id]
    seg_ids = range(first_section_start[0], first_section_end[0])

    seg_lengths = get_segment_lengths(self, seg_ids)
    return seg_lengths.sum()



def get_branch_order(tree, seg_id):
    """Return branch order of segment."""
    B = tree.get_multifurcations()
    return sum(1 if i in B else 0 for i in get_way_to_root(tree, seg_id))


def get_point_section_branch_orders(self):
    """Tree method to get section lengths."""
    return np.array([get_branch_order(self, i) for i in range(size(self))])


def get_point_projection(self, vect=(0, 1, 0), point=None):
    """Projects each point in the tree (x,y,z) - input_point to a selected vector.

    This gives the orientation of
    each section according to a vector in space, if normalized,
    otherwise it return the relative length of the section.
    """
    if point is None:
        point = [self.x[0], self.y[0], self.z[0]]

    xyz = np.transpose([self.x, self.y, self.z]) - point

    return np.dot(xyz, vect)




def get_direction_between(self, start_id=0, end_id=1):
    """Return direction of a branch.

    The direction is defined as end point - start point normalized as a unit vector.
    """
    # pylint: disable=assignment-from-no-return
    vect = np.subtract(
        [self.x[end_id], self.y[end_id], self.z[end_id]],
        [self.x[start_id], self.y[start_id], self.z[start_id]],
    )

    if np.linalg.norm(vect) != 0.0:
        return vect / np.linalg.norm(vect)
    return vect


def _vec_angle(u, v):
    """Return the angle between v and u in 3D."""
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    return np.arccos(c)


def get_angle_between(tree, sec_id1, sec_id2):  # noqa: D417
    """Return local bifurcations angle between two sections, defined by their ids.

    Args:
        sec_id1: the start point of the section #1
        sec_id2: the start point of the section #2
    """
    beg, end = tree.get_sections_only_points()
    b1 = np.where(beg == sec_id1)[0][0]
    b2 = np.where(beg == sec_id2)[0][0]

    u = tree.get_direction_between(beg[b1], end[b1])
    v = tree.get_direction_between(beg[b2], end[b2])

    return _vec_angle(u, v)



def get_pca(self, plane="xy", component=0):
    """Return the i-th principal component of PCA of the tree points in the selected plane."""
    pca = PCA(n_components=2)
    pca.fit(np.transpose([getattr(self, plane[0]), getattr(self, plane[1])]))

    return pca.components_[component]
