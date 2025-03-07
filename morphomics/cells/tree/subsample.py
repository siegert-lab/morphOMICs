import numpy as np
from scipy.stats import geom

def prune_branch(self, leaf, nb_nodes):
    way = self.get_way_to_root(node_idx=leaf)
    way = way[::-1]
    if isinstance(nb_nodes, int):
        cut = nb_nodes
    else:
        cut = geom.rvs(p=1-nb_nodes) - 1
    if cut > 0:
        way = way[:-cut]
    return way

def cut_branch(self, leaf, degree):
    child_parent_map = self.parents_children()[0]
    parent_degree = leaf
    intermediate_nodes = [parent_degree]
    while parent_degree != -1:
        parent_degree = child_parent_map[parent_degree]
        intermediate_nodes.append(parent_degree)

    if len(intermediate_nodes) < degree+2:
        new_leaf = intermediate_nodes[0]
    else:
        new_leaf = intermediate_nodes[-degree-2]

    way = self.get_way_to_root(node_idx = new_leaf)
    way = way[::-1]
    return way
