import numpy as np
from morphomics.cells.neuron.neuron import TREE_TYPE_TO_DIGIT 
from morphomics.io.swc import SWC_DCT

def check_unique_soma_on_first_line(swc):
    swc = swc.T
    soma_index = TREE_TYPE_TO_DIGIT["soma"]
    soma_ids = np.where(swc[SWC_DCT["type"]] == soma_index)[0]
    if len(soma_ids) != 1:
        return False
    if soma_ids[0] != 0:
        return False
    if swc[SWC_DCT["index"], soma_ids[0]] != 1:
        return False
    
    return True

def check_num_soma_children(swc):
    return (swc[:, SWC_DCT["parent"]]==1).sum()

def check_num_neurites(cell):
    return len(cell.neurites)