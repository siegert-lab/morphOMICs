"""TMD Topology algorithms implementation."""

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
import scipy.spatial as sp

#from morphomics.tmd.analysis import sort_ph
from morphomics.persistent_homology.persistent_properties import NoProperty
from morphomics.persistent_homology.persistent_properties import PersistentAngles
from morphomics.persistent_homology.persistent_properties import PersistentMeanRadius


def tree_to_property_barcode(tree, filtration_function, property_class=NoProperty):
    """Decompose a tree data structure into a barcode.

    Each bar in the barcode is optionally linked with a property determined by property_class.

    Args:
        filtration_function (Callable[tree] -> np.ndarray):
            The filtration function to apply on the tree

        property_class (PersistentProperty, optional): A PersistentProperty class.By
            default the NoProperty is used which does not add entries in the barcode.

    Returns:
        barcode (list): A list of bars [bar1, bar2, ..., barN], where each bar is a
            list of:
                - filtration value start
                - filtration value end
                - property_value1
                - property_value2
                - ...
                - property_valueN
        bars_to_points: A list of point ids for each bar in the barcode. Each list
            corresponds to the set of endpoints (i.e. the end point of each section)
            that belong to the corresponding persistent component - or bar.
    """
    point_values = filtration_function(tree)

    beg, _ = tree.sections
    parents, children = tree.parents_children()

    prop = property_class(tree)

    active = tree.get_node_children_number() == 0
    alives = np.where(active)[0]
    point_ids_track = {al: [al] for al in alives}
    bars_to_points = []

    ph = []
    while len(alives) > 1:
        for alive in alives:
            p = parents[alive]
            c = children[p]

            if np.all(active[c]):
                active[p] = True
                active[c] = False

                mx = np.argmax(abs(point_values[c]))
                mx_id = c[mx]

                c = np.delete(c, mx)

                for ci in c:
                    component_id = np.where(beg == p)[0][0]
                    ph.append([point_values[ci], point_values[p]] + prop.get(component_id))
                    bars_to_points.append(point_ids_track[ci])

                point_values[p] = point_values[mx_id]
                point_ids_track[p] = point_ids_track[mx_id] + [p]
        alives = np.where(active)[0]

    ph.append(
        [point_values[alives[0]], 0] + prop.infinite_component(beg[0])
    )  # Add the last alive component
    bars_to_points.append(point_ids_track[alives[0]])
    ph = np.array(ph)
    return ph, bars_to_points


def _filtration_function(feature, **kwargs):
    """Returns filtration function lambda that will be applied point-wise on the tree."""
    return lambda tree: getattr(tree, "get_nodes_" + feature)(**kwargs)


def get_persistence_diagram(tree, feature="radial_distance", **kwargs):
    """Method to extract ph from tree that contains mutlifurcations."""
    ph, _ = tree_to_property_barcode(
        tree, filtration_function=_filtration_function(feature, **kwargs), 
        property_class=NoProperty
    )
    return ph


def get_ph_angles(tree, feature="radial_distance", **kwargs):
    """Method to extract ph from tree that contains mutlifurcations."""
    ph, _ = tree_to_property_barcode(
        tree,
        filtration_function=_filtration_function(feature, **kwargs),
        property_class=PersistentAngles,
    )
    return ph


def get_ph_radii(tree, feature="radial_distance", **kwargs):
    """Returns the ph diagram enhanced with the corresponding encoded radii."""
    ph, _ = tree_to_property_barcode(
        tree,
        filtration_function=_filtration_function(feature, **kwargs),
        property_class=PersistentMeanRadius,
    )
    return ph


def get_ph_neuron(neuron, feature="radial_distance", neurite_type="all", **kwargs):
    """Method to extract ph from a neuron that contains mutlifurcations."""
    ph_neuron = []

    if neurite_type == "all":
        type_list = ["neurites"]
    else:
        type_list = [neurite_type]

    for type in type_list:
        for tree in getattr(neuron, type):
            ph_tree = get_persistence_diagram(tree, feature=feature, **kwargs)
            ph_neuron.append(ph_tree)

    ph_neuron = np.vstack(ph_neuron)
    return ph_neuron






