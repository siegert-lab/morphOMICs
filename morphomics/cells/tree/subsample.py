import numpy as np
from scipy.stats import geom

def prune_branch(self, leaf, nb_nodes):
    way = self.get_way_to_root(sec_id=leaf)
    way = way[::-1]
    way.append(leaf)
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
    for _ in range(degree):
        parent_degree = child_parent_map[parent_degree]
    way = self.get_way_to_root(sec_id = parent_degree)
    way = way[::-1]
    way.append(parent_degree)
    return way
