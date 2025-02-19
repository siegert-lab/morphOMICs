import warnings
import numpy as np
import pandas as pd
import os
from morphomics.io.io import read_swc, get_info_frame
from morphomics.io.swc import swc_to_neuron
from morphomics.persistent_homology.tmd import get_ph_neuron, get_persistence_diagram

class Population:
    """A Population object is a container for Neurons.

    Args:
        name (str): The name of the Population.
        neurons (list[tmd.Neuron.Neuron.Neuron]): A list of neurons to include in the Population.
    """
    def __init__(self, info_frame = None,
               cells_frame = None, 
               folder_path = None,
               name = None, 
               conditions = {'Region' : None,
                             'Condition' : None,
                             'Model' : None,
                             'Time' : None,
                             'Sex' : None,
                             'Animal' : None}
                ):
        """
        Initialize the swc array and the Neuron instance for each sample in the DataFrame.
        """
        if name is None and folder_path is not None:
            self.name =  os.path.basename(folder_path)
        self.folder_path = folder_path
        self.conditions = conditions

        if isinstance(self.conditions, dict):
            conds = list(self.conditions.keys())
        else:
            conds = list(self.conditions)

        if self.folder_path is not None:
            info_frame = get_info_frame(self.folder_path,
                        extension = ".swc",
                        conditions = conds)

        # Read infoframe and add a column for cells graph from swc files.
        self.cells = pd.DataFrame()
        if info_frame is not None:
            assert (
                "file_path" in info_frame.keys()
            ), "`file_path` must be a column in the info_frame DataFrame"
            morphoframe = {}
            info_frame_parts = np.array_split(info_frame, 4)
            for i, sub_info_frame in enumerate(info_frame_parts):
                # Read the swc files and add them in the column swc_array.
                sub_info_frame['swc_array'] = sub_info_frame['file_path'].apply(lambda file_path: read_swc(file_path))                        
                sub_info_frame['cells'] = sub_info_frame['swc_array'].apply(lambda swc_arr: swc_to_neuron(swc_arr) if swc_arr is not np.nan else np.nan)
                print("You have loaded %d%% chunk of the data..."%(((i+1)/4)*100))
                morphoframe[str(i)] = sub_info_frame
   
            self.cells = pd.concat([sub_frame for sub_frame in morphoframe.values()], ignore_index=True)
            print(" ")

        # Add cells already in a Panda DataFrame                                            
        if cells_frame is not None:
            self.cells = pd.concat((self.cells, cells_frame))

        self.set_empty_cells_to_nan()
    
    def exclude_sg_branches(self):
        """
        Remove from each cell the branches that do not have ramification.
        I.e. the single trunks. 
        """
        self.cells['cells'].apply(lambda cell: cell.exclude_small_trees(nb_sections=1)
                                                                   if cell is not np.nan else np.nan
                                                                )
        self.set_empty_cells_to_nan()
        
    def set_barcodes(self, filtration_function = 'radial_distance', from_trees = True):
        """
        Calculates persistence diagram of each cell graph.
        """
        assert filtration_function in ["radial_distance",
                            "path_distance",
                            ], "Currently, TMD is only implemented with either radial_distance or path_distance"
        
        if from_trees:
            if 'trees' not in self.cells.keys():
                self.combine_neurites()
            self.cells['barcodes'] = self.cells['trees'].apply(lambda tree: get_persistence_diagram(tree = tree, feature=filtration_function
                                                                                                ) if tree is not np.nan else np.nan
                                                                )
        else:
            self.cells['barcodes'] = self.cells['cells'].apply(lambda cell: get_ph_neuron(neuron = cell, feature=filtration_function
                                                                                                ) if cell is not np.nan else np.nan
                                                                )
    
    def simplify(self):
        self.cells['simplified_cells'] = self.cells['cells'].apply(lambda cell: cell.simplify()
                                                                   if cell is not np.nan else np.nan
                                                                   )
        
    def combine_neurites(self):
        self.cells['trees'] = self.cells['cells'].apply(lambda cell: cell.combine_neurites().neurites[0]
                                                                   if cell is not np.nan else np.nan
                                                                   )
        
    def apply_tree_method(self, method, col_name=None, **kwargs):
        if 'trees' not in self.cells.keys():
            self.combine_neurites()
        if col_name is None:
            col_name = method
       
        if method == 'move_to_point':  # This is the only method of the class Tree that is inplace
            self.cells['trees'].apply(lambda tree: tree.move_to_point( **kwargs)
                                                        if tree is not np.nan else np.nan
                                                        )
        else:
            self.cells[col_name] = self.cells['trees'].apply(lambda tree: getattr(tree, method)( **kwargs)
                                                            if tree is not np.nan else np.nan
                                                            )

    def set_empty_cells_to_nan(self):
        # Convert Neuron with empty list of Trees into np.nan.
        self.cells['cells'] = self.cells['cells'].apply(lambda cell: np.nan if cell is not np.nan and isinstance(cell.neurites, list) and len(cell.neurites) == 0 else cell)
    