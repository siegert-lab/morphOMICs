import warnings
import numpy as np
import pandas as pd
import os
from morphomics.io.io import read_swc
from morphomics.io.swc import swc_to_neuron
from morphomics.persistent_homology.tmd import get_ph_neuron

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
            self.set_empty_cells_to_nan()

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
        
    def set_barcodes(self, filtration_function = 'radial_distance'):
        """
        Calculates persistence diagram of each cell graph.
        """
        assert filtration_function in ["radial_distance",
                            "path_distance",
                            ], "Currently, TMD is only implemented with either radial_distance or path_distance"

        # Get a column composed of Neuron instances.
        self.cells['barcodes'] = self.cells['cells'].apply(lambda cell: get_ph_neuron(neuron = cell, feature=filtration_function
                                                                                                ) if cell is not np.nan else np.nan
                                                                )

    def get_section_length(self):
        """
        Get the length of all the section of branches for each cell.
        """
        self.cells['section_lengths'] = self.cells['cells'].apply(lambda cell: np.concatenate([tree.get_sections_length() for tree in cell.neurites]
                                                                                            ) if cell is not np.nan else np.nan)

    def set_empty_cells_to_nan(self):
        # Convert Neuron with empty list of Trees into np.nan.
        self.cells['cells'] = self.cells['cells'].apply(lambda cell: np.nan if cell is not np.nan and isinstance(cell.neurites, list) and len(cell.neurites) == 0 else cell)
    