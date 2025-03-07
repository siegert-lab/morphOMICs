"""TMD Neuron's methods."""

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
from collections import Counter

def get_nb_trees(self, tree_type_list="all"):
    """Neuron method to get the number of tress."""
    if tree_type_list == "all":
        tree_type_list = ["basal_dendrite", "axon", "apical_dendrite", "glia_process", "undefined"]
    s = np.sum([len(getattr(self, tree_type)) for tree_type in tree_type_list])
    return int(s)

def get_bounding_box(self):
    """Get the bounding box of the neurites.

    Args:
        neuron: A TMD neuron.

    Returns:
        bounding_box: np.array
            ([xmin,ymin,zmin], [xmax,ymax,zmax])
    """
    x = []
    y = []
    z = []

    for tree in self.neurites:
        x = x + tree.x.tolist()
        y = y + tree.y.tolist()
        z = z + tree.z.tolist()

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)

    return np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])

def get_neurites_type(self):
    type_list = [neurite.get_type() for neurite in self.neurites]
    # Count occurrences of each digit in the list
    digit_counts = Counter(type_list)
    # Create a dictionary with neurite types as keys and their counts as values
    neurite_counts = {self.tree_type_dict[digit]: count for digit, count in digit_counts.items()}
    return neurite_counts

