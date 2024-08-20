import copy
import numpy as np
from collections import OrderedDict
from itertools import starmap
from morphomics.utils import distances
import scipy.sparse as sp

def size(self):
    """Tree method to get the size of the tree list.
        i.e the number of nodes in the graph (tree).
    """
    return int(len(self.x))

def get_bounding_box(self):
    """Get the bounding box of the neurites.

    Args:
        self: A TMD tree.

    Return:
        bounding_box: np.array
            ([xmin,ymin,zmin], [xmax,ymax,zmax])
    """
    xmin = np.min(self.x)
    xmax = np.max(self.x)
    ymin = np.min(self.y)
    ymax = np.max(self.y)
    zmin = np.min(self.z)
    zmax = np.max(self.z)

    return np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])

def get_type(self):
    """Return type of tree."""
    return int(np.median(self.t))

# Segment features
def get_edges_coords(self, seg_ids=None):
    """Return edges coordinates.

    Args:
        self: A TMD tree.
        seg_ids: segment numbers to consider

    Return:
        seg_list: np.array
            (child[x,y,z], parent[x,y,z])
    """
    seg_list = []
    if not seg_ids:
        seg_ids = range(0, self.size() - 1)

    for seg_id in seg_ids:
        par_id = self.p[seg_id + 1]
        child_coords = np.array([self.x[seg_id + 1], self.y[seg_id + 1], self.z[seg_id + 1]])
        parent_coords = np.array([self.x[par_id], self.y[par_id], self.z[par_id]])
        seg_list.append(np.array([parent_coords, child_coords]))

    return seg_list

def get_edges_length(self, seg_ids=None):
    """Return edges lengths.

    Args:
        tree: tmd tree
        seg_ids: segment numbers to consider
    """
    if not seg_ids:
        seg_ids = range(0, self.size() - 1)

    segs = self.get_edges_coords(seg_ids)

    seg_len = np.fromiter(starmap(distances['l2'], segs), dtype=float)

    return seg_len

# Points features to be used for topological extraction
def get_nodes_radial_distance(self, point=None, dim="xyz"):
    """Tree method to get radial distances from nodes in a Tree.

    If point is None, the soma surface -defined by
    the initial point of the tree- will be used
    as a reference point.
    """
    if point is None:
        point = []
        for d in dim:
            point.append(getattr(self, d)[0])

    radial_distances = np.zeros(self.size(), dtype=float)

    for i in range(self.size()):
        point_dest = []
        for d in dim:
            point_dest.append(getattr(self, d)[i])

        radial_distances[i] = distances['l2'](point, point_dest)

    return radial_distances

def get_children(self):
    """Return a dictionary of children for each node of the tree."""
    return OrderedDict({i: np.where(self.p == i)[0] for i in range(len(self.p))})

def get_bif_term(self):
    """Return number of children per point."""
    return np.array(sp.csr_matrix.sum(self.dA, axis=0))[0]

def get_nodes_path_distance(self):
    """Tree method to get path distances from the root."""
    edge_len = get_edges_length(self)
    path_lengths = np.append(0, copy.deepcopy(edge_len))
    children = get_children(self)

    for k, v in children.items():
        path_lengths[v] = path_lengths[v] + path_lengths[k]

    return path_lengths

def get_lifetime(self, feature="nodes_radial_distance"):
    """Returns the sequence of birth - death times for each section.

    This can be used as the first step for the approximation of P.H.
    of the radial distances of the neuronal branches.
    """
    begs, ends = self.sections
    rd = getattr(self, "get_" + feature)()
    lifetime = np.array(len(begs) * [np.zeros(2)])

    for i, (beg, end) in enumerate(zip(begs, ends)):
        lifetime[i] = np.array([rd[beg], rd[end]])

    return lifetime