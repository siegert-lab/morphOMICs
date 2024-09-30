"""TMD class : Neuron."""

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
import warnings

import numpy as np

from morphomics.cells.soma.soma import Soma
from morphomics.cells.tree.tree import Tree

TREE_TYPE_DICT = {1: "soma", 2: "axon", 3: "basal_dendrite", 4: "apical_dendrite", 5: "process"}


class Neuron:
    """A Neuron object is a container for Trees and a Soma.

    The Trees can be basal_dendrite, apical_dendrite and axon.

    Args:
        name (str): The name of the Neuron.
    """

    # pylint: disable=import-outside-toplevel
    from morphomics.cells.neuron.methods import get_bounding_box
    from morphomics.cells.neuron.methods import size

    def __init__(self, name="Neuron"):
        """Creates an empty Neuron object."""
        self.soma = Soma()
        self.axon = []
        self.apical_dendrite = []
        self.basal_dendrite = []
        self.process = []
        self.undefined = []
        self.name = name

    @property
    def neurites(self):
        """Get neurites."""
        return self.apical_dendrite + self.axon + self.basal_dendrite + self.process

    @property
    def dendrites(self):
        """Get dendrites."""
        return self.apical_dendrite + self.basal_dendrite

    def rename(self, new_name):
        """Modifies the name of the Neuron to new_name."""
        self.name = new_name

    def set_soma(self, new_soma):
        """Set the given Soma object as the soma of the current Neuron."""
        if isinstance(new_soma, Soma):
            self.soma = new_soma

    def append_tree(self, new_tree, tree_types):
        """Append a Tree object to the Neuron.

        If type of object is tree this function finds the type of tree and adds the new_tree to the
        correct list of trees in neuron.
        """
        if isinstance(new_tree, Tree):
            t_type = new_tree.get_type()
            if t_type in tree_types.keys():
                neurite_type = tree_types[t_type]
            else:
                neurite_type = "undefined"
            getattr(self, neurite_type).append(new_tree)

    def remove_tree(self, tree_type, tree):
        cell_type = getattr(self, tree_type, None)
        try:
            cell_type.remove(tree)
        except ValueError:
            print(f"value not found in the list")

    def exclude_small_branches(self, nb_sections = 1):
        for i, tree in enumerate(self.neurites):
            beg, _ = tree.sections
            if len(beg) <= nb_sections:
                t = tree.get_type()
                t = TREE_TYPE_DICT[t]
                self.remove_tree(tree_type = t, tree = tree)

    def copy_neuron(self):
        """Returns a deep copy of the Neuron."""
        return copy.deepcopy(self)

    def simplify(self):
        """Creates a copy of itself and simplifies all trees to create a skeleton of the neuron."""
        neu = Neuron()
        neu.soma = self.soma.copy_soma()

        for tr in self.neurites:
            t = tr.simplify()
            neu.append_tree(t, TREE_TYPE_DICT)

        return neu
    
    def combine_neurites(self):
        # Set soma as the root
        x_list, y_list, z_list, d_list, t_list, p_list = [self.soma.x], [self.soma.y], [self.soma.z], [self.soma.d], [1], [-1]

        nb_nodes = 1
        for tr in self.neurites:
            x_list.append(tr.x)
            y_list.append(tr.y)
            z_list.append(tr.z)
            d_list.append(tr.d)
            t_list.append(tr.t)
            
            # Adjust the parent indices for all but the first tree
            adjusted_p = np.hstack([tr.p[0] + 1, tr.p[1:] + nb_nodes])
            p_list.append(adjusted_p)
            
            nb_nodes += len(tr.p)

        # Horizontally stack each list into a single NumPy array
        x_array = np.hstack(x_list)
        y_array = np.hstack(y_list)
        z_array = np.hstack(z_list)
        d_array = np.hstack(d_list)
        t_array = np.hstack(t_list)
        p_array = np.hstack(p_list)

        new_tree = Tree(x_array, y_array, z_array, d_array, t_array, p_array)
        
        neu = Neuron()
        neu.append_tree(new_tree, TREE_TYPE_DICT)

        return neu
    



