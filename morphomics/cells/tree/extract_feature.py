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

# Connectivity features
def get_children(self):
    """Return a dictionary of children for each node of the tree."""
    return OrderedDict({i: np.where(self.p == i)[0] for i in range(len(self.p))})

def get_node_children_number(self):
    """Return number of children per node."""
    return np.array(sp.csr_matrix.sum(self.dA, axis=0))[0]

def get_bifurcations(self):
    """Return nodes index that has exactly two children."""
    bif_term = self.get_node_children_number()
    bif = np.where(bif_term == 2.0)[0]
    return bif

def get_multifurcations(self):
    """Return bifurcations."""
    bif_term = self.get_node_children_number()
    bif = np.where(bif_term >= 2.0)[0]
    return bif

def get_terminations(self):
    """Return terminations."""
    bif_term = self.get_node_children_number()
    term = np.where(bif_term == 0.0)[0]
    return term

def get_way_to_root(self, sec_id=0):
    """Return way to root. 
    It returns a list of parented nodes from the input node to the root."""
    way = [sec_id]
    tmp_id = sec_id

    while tmp_id != -1:
        way.append(self.p[tmp_id])
        tmp_id = self.p[tmp_id]
    # remove -1 at the end of way
    way.pop()
    return way

def get_way_order(self, seg_id):
    """Return the number of multibifurcation nodes on the way."""
    multibif_ids = self.get_multifurcations()
    way_to_root = self.get_way_to_root(seg_id)
    intersection = set(multibif_ids).intersection(way_to_root)
    return len(intersection)

def get_nodes_way_order(self):
    """Return the list of way order.
     This can be also interpreted as an approximation of the number
      of sections between nodes and root."""
    return np.array([self.get_way_order(i) for i in range(self.size())])

# Edges features
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

def get_sections_length(self):
    """Tree method to get section lengths."""
    lengths = np.zeros(self.size(), dtype=float)
    ways, end = self.sections
    edge_len = self.get_edges_length()

    for start_id, end_id in zip(ways, end):
        lengths[end_id] = np.sum(edge_len[max(0, start_id - 1) : end_id])

    return lengths

# Angles
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

def get_angle_between(self, sec_id1, sec_id2):  # noqa: D417
    """Return local bifurcations angle between two sections, defined by their ids.

    Args:
        sec_id1: the start point of the section #1
        sec_id2: the start point of the section #2
    """
    beg, end = self.sections
    b1 = np.where(beg == sec_id1)[0][0]
    b2 = np.where(beg == sec_id2)[0][0]

    u = self.get_direction_between(beg[b1], end[b1])
    v = self.get_direction_between(beg[b2], end[b2])

    return _vec_angle(u, v)

# Nodes features to be used for topological extraction
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

def get_nodes_path_distance(self):
    """Tree method to get path distances from the root."""
    edge_len = self.get_edges_length()
    path_lengths = np.append(0, copy.deepcopy(edge_len))
    children = self.get_children()

    for k, v in children.items():
        path_lengths[v] = path_lengths[v] + path_lengths[k]

    return path_lengths


