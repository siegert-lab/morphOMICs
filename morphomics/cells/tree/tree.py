"""TMD class : Tree."""

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

import numpy as np
import scipy.sparse as sp
from cached_property import cached_property
from functools import lru_cache

class Tree:
    """Tree class.

    Args:
        x (list[float]): The x-coordinates of the tree segments.
        y (list[float]): The y-coordinates of the tree segments.
        z (list[float]): The z-coordinate of the tree segments.
        d (list[float]): The diameters of the tree segments.
        t (list[int]): The types (basal_dendrite, apical_dendrite, axon) of the tree segments.
        p (list[int]): The index of the parent of the tree segments.
    """

    # pylint: disable=import-outside-toplevel
    from morphomics.cells.tree.extract_feature import size
    from morphomics.cells.tree.extract_feature import get_bounding_box
    from morphomics.cells.tree.extract_feature import get_type

    from morphomics.cells.tree.extract_feature import get_children
    from morphomics.cells.tree.extract_feature import get_node_children_number
    from morphomics.cells.tree.extract_feature import get_bifurcations
    from morphomics.cells.tree.extract_feature import get_multifurcations
    from morphomics.cells.tree.extract_feature import get_terminations

    from morphomics.cells.tree.extract_feature import get_way_to_root
    from morphomics.cells.tree.extract_feature import get_way_length
    from morphomics.cells.tree.extract_feature import get_way_order
    from morphomics.cells.tree.extract_feature import get_nodes_way_order
    
    from morphomics.cells.tree.extract_feature import get_edges_coords
    from morphomics.cells.tree.extract_feature import get_edges_length
    from morphomics.cells.tree.extract_feature import get_lifetime
    # from morphomics.cells.tree.extract_feature import get_sections_only_points
    from morphomics.cells.tree.extract_feature import get_sections_length
    
    from morphomics.cells.tree.extract_feature import get_direction_between
    from morphomics.cells.tree.extract_feature import _vec_angle
    from morphomics.cells.tree.extract_feature import get_angle_between_sections

    from morphomics.cells.tree.extract_feature import get_nodes_radial_distance
    from morphomics.cells.tree.extract_feature import get_nodes_path_distance

    from morphomics.cells.tree.subsample import prune_branch
    from morphomics.cells.tree.subsample import cut_branch

    def __init__(self, x, y, z, d, t, p):
        """Constructor of tmd Tree Object."""
        try:
            self.x = np.array(x, dtype=np.float32)
            self.y = np.array(y, dtype=np.float32)
            self.z = np.array(z, dtype=np.float32)
            self.d = np.array(d, dtype=np.float32)
            self.t = np.array(t, dtype=np.int32)
            self.p = np.array(p, dtype=np.int64)
            # Check if all arrays have the same length
            lengths = [len(self.x), len(self.y), len(self.z), len(self.d), len(self.t), len(self.p)]
            if not all(length == lengths[0] for length in lengths):
                raise ValueError("All input arrays must have the same length.")
        except ValueError as e:
            print(e)

        self.dA = sp.csr_matrix(
            (np.ones(len(self.x) - 1), (range(1, len(self.x)), self.p[1:])),
            shape=(len(self.x), len(self.x)),
        )

    def copy_tree(self):
        """Returns a deep copy of the Tree."""
        return copy.deepcopy(self)
    
    def is_equal(self, tree):
        """Tests if all tree lists are the same."""
        eq = np.all(
            [
                np.allclose(self.x, tree.x, atol=1e-4),
                np.allclose(self.y, tree.y, atol=1e-4),
                np.allclose(self.z, tree.z, atol=1e-4),
                np.allclose(self.d, tree.d, atol=1e-4),
                np.allclose(self.t, tree.t, atol=1e-4),
                np.allclose(self.p, tree.p, atol=1e-4),
            ]
        )
        return eq
    
    @cached_property
    def edges(self):
        """Returns the parents and children nodes of each edge.

        Returns:
            tuple:
                parents (np.ndarray):
                    The starting point ids of edges
                children (np.ndarray)
                    The ending point ids of edges
        """
        parents = self.p[1:]
        children = np.array(range(1, self.size()))        
        
        return parents, children
    
    @cached_property
    def sections(self):
        """Get the sections boundaries of the current tree.
        A section is a subset of the tree where each node has only one child, excepeted the boundaries.
        The beg boundary has more than one child and the end boundary has no child.

        Returns:
            tuple:
                section_beg_point_ids (np.ndarray):
                    The starting point ids of sections
                section_end_point_ids (np.ndarray)
                    The ending point ids of sections
        """
        """Get indices of the parents of the first sections' points and of their last points."""
        children_counts = self.get_node_children_number() 
        end = np.array(children_counts != 1).nonzero()[0]
        if 0 in end:  # If first segment is a bifurcation
            end = end[1:]

        beg = np.append([0], self.p[np.delete(np.hstack([0, 1 + end]), len(end))][1:])
        return beg, end
    
    @lru_cache(maxsize=None)  # This decorator caches the results
    def parents_children(self, edges=False):
        """Returns the dictionnaries of parents to children and children to parents.

        Returns:
            children_to_parents (dict): Each key corresponds to a section id (node)
                and the respective values to the parent section ids (node).
            parents_to_children (dict): Each key corresponds to a section id (node)
                and the respective values to the children section ids (nodes)

        Notes:
            If 0 exists in starting nodes, the parent from tree is assigned
        """
        if len(self.p) == 1:
            return {}, {}
        if edges:
            begs, ends = self.edges
        else:
            begs, ends = self.sections

        children_to_parents = {e: b for b, e in zip(begs, ends)}

        if 0 in begs:
            children_to_parents[0] = self.p[0]

        parents_to_children = {b: ends[np.where(begs == b)[0]] for b in np.unique(begs)}

        return children_to_parents, parents_to_children
    
    def move_to_point(self, point=(0, 0, 0)):
        """Moves the tree in the x-y-z plane so that it starts from the selected point."""
        self.x = self.x - (self.x[0]) + point[0]
        self.y = self.y - (self.y[0]) + point[1]
        self.z = self.z - (self.z[0]) + point[2]

    def rotate_xy(self, angle):
        """Returns a rotated tree in the x-y plane by the defined angle."""
        new_x = self.x * np.cos(angle) - self.y * np.sin(angle)
        new_y = self.x * np.sin(angle) + self.y * np.cos(angle)

        return Tree(new_x, new_y, self.z, self.d, self.t, self.p)

    def simplify(self):
        """Returns a simplified tree that corresponds to the start - end of the sections points."""
        if self.size() == 1:
            return self
         
        beg0, end0 = self.sections
        sections = np.transpose([beg0, end0])

        x = np.zeros([len(sections) + 1])
        y = np.zeros([len(sections) + 1])
        z = np.zeros([len(sections) + 1])
        d = np.zeros([len(sections) + 1])
        t = np.zeros([len(sections) + 1])
        p = np.zeros([len(sections) + 1])

        x[0] = self.x[sections[0][0]]
        y[0] = self.y[sections[0][0]]
        z[0] = self.z[sections[0][0]]
        d[0] = self.d[sections[0][0]]
        t[0] = self.t[sections[0][0]]
        p[0] = -1

        for i, s in enumerate(sections):
            x[i + 1] = self.x[s[1]]
            y[i + 1] = self.y[s[1]]
            z[i + 1] = self.z[s[1]]
            d[i + 1] = self.d[s[1]]
            t[i + 1] = self.t[s[1]]
            p[i + 1] = np.where(beg0 == s[0])[0][0]

        return Tree(x, y, z, d, t, p)
    
    # def simplify(self):
    #     """Returns a simplified tree that corresponds to the start - end of the sections points."""
    #     k_out = self.get_node_children_number()

    def subsample_tree(self, _type, number):
        tip_starts = self.get_terminations()
        subsampled_nodes = set()

        for leaf in tip_starts:
            if _type == 'cut':
                way = self.cut_branch(leaf, degree = number)
            elif _type == 'prune':
                way = self.prune_branch(leaf, nb_nodes = number)
            subsampled_nodes.update(way)
        subsampled_nodes = list(subsampled_nodes)
        if len(subsampled_nodes) == 0:
            subsampled_nodes = [0]
        new_x = self.x[subsampled_nodes]
        new_y = self.y[subsampled_nodes]
        new_z = self.z[subsampled_nodes]
        new_d = self.d[subsampled_nodes]
        new_t = self.t[subsampled_nodes]
        new_p = self.p[subsampled_nodes]

        # Step 1: Create a mapping of nodes to their indices in subsampled_nodes
        index_map = {node: index for index, node in enumerate(subsampled_nodes)}
        index_map[-1] = -1
        # Step 2: Replace each number in list p with its index from subsampled_nodes
        new_p = [index_map[number] for number in new_p]

        if len(subsampled_nodes) > 1:
            new_tree = Tree(new_x, new_y, new_z, new_d, new_t, new_p)
            return new_tree
        else:
            return None