import warnings
import numpy as np

from morphomics.cells.neuron.neuron import Neuron
from morphomics.cells.soma.soma import Soma
from morphomics.cells.tree.tree import Tree

from scipy import sparse as sp
from scipy.sparse import csgraph as cs
from morphomics.cells.neuron.neuron import DIGIT_TO_TREE_TYPE, TREE_TYPE_TO_DIGIT 
from morphomics.cells.utils import LoadNeuronError

# Definition of swc data container
SWC_DCT = {"index": 0, "type": 1, "x": 2, "y": 3, "z": 4, "radius": 5, "parent": 6}

def swc_to_tree(swc_arr):
    """Make tree structure from loaded data."""
    tr_data = np.transpose(swc_arr)

    parents = [
        np.where(tr_data[SWC_DCT["index"]] == i)[0][0] if len(np.where(tr_data[0] == i)[0]) > 0 else -1
        for i in tr_data[SWC_DCT["parent"]]
    ]

    return Tree(
        x=tr_data[SWC_DCT["x"]],
        y=tr_data[SWC_DCT["y"]],
        z=tr_data[SWC_DCT["z"]],
        d=tr_data[SWC_DCT["radius"]],
        t=tr_data[SWC_DCT["type"]],
        p=parents,
    )


def swc_to_neuron(swc_arr, name = 'Microglia'):
    # Definition of swc types from type_dict function
    swc_arr_T = np.transpose(swc_arr)

    neuron = Neuron(name = name)
    
    # Check for duplicated IDs
    IDs_counts = np.unique(swc_arr_T[SWC_DCT['index']], return_counts=True)
    IDs, counts = IDs_counts[0], IDs_counts[1]
    if (counts != 1).any():
        warnings.warn(f"The following IDs are duplicated: {IDs[counts > 1]}")

    # Check the ID of the soma
    soma_digit = TREE_TYPE_TO_DIGIT["soma"]
    try:
        soma_ids = np.where(swc_arr_T[SWC_DCT["type"]] == soma_digit)[0]
    except IndexError:
        raise LoadNeuronError("Soma points not in the expected format")
    
    # Extract soma information from swc
    soma = Soma(
        x = swc_arr_T[SWC_DCT["x"]][soma_ids],
        y = swc_arr_T[SWC_DCT["y"]][soma_ids],
        z = swc_arr_T[SWC_DCT["z"]][soma_ids],
        d = swc_arr_T[SWC_DCT["radius"]][soma_ids],
    )

    # Save soma in Neuron
    neuron.set_soma(soma)
    
    p = np.array(swc_arr_T[SWC_DCT["parent"]], dtype=int) - swc_arr_T[0][0]
    
    # return p, soma_ids
    try:
        dA = sp.csr_matrix(
            (
                # This is the weight of a link 
                np.ones(len(p) - len(soma_ids)),
                # This is the list of row (child) and columns (parents)
                (range(len(soma_ids), len(p)), p[len(soma_ids) :]),
            ),
            shape=(len(p), len(p)),
        )
    except Exception:
        raise LoadNeuronError(
            "Cannot create connectivity, nodes not connected correctly."
        )

    # assuming soma points are in the beginning of the file.
    comp = cs.connected_components(dA[len(soma_ids) :, len(soma_ids) :])

    # Extract trees
    for i in range(comp[0]):
        tree_ids = np.where(comp[1] == i)[0] + len(soma_ids)
        tree = swc_to_tree(swc_arr[tree_ids])
        neuron.append_tree(tree)

    return neuron