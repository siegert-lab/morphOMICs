import os

import morphomics
from morphomics.io import io

from morphomics.cells.population.population import Population

from morphomics.protocols.default_parameters import DefaultParams
from morphomics.protocols import subsampler
from morphomics.protocols.vectorizer import Vectorizer
from morphomics.protocols.dim_reducer import DimReducer
from morphomics.protocols import plotting

from morphomics.utils import save_obj, load_obj, vectorization_codenames
from sklearn.preprocessing import Normalizer, StandardScaler

import numpy as np
import pandas as pd


class Pipeline(object):
    
    def __init__(self, parameters, Parameters_ID, morphoframe = {}, metadata = {}) :
        """
        Initializes the Pipeline instance.

        Parameters
        ----------
        parameters (dict): Contains the parameters for each protocol that would be run.
                            example:
                            parameters = {'protocol_1 : { parameter_1_1: x_1_1, ..., parameter_1_n: x_1_n},
                                            ...
                                            'protocol_m : { parameter_m_1: x_m_1, ..., parameter_m_k: x_m_k}
                                        } 
        Parameters_ID (str): A way to characterize the name of all saved files from the same Pipeline instance i.e. prefix.
        morphoframe (dict): Contains the data from cells, each column is a feature and each row is a cell.
        metadata (dict): Contains the tools that the pipeline use, contains the data that are not ordered by row/cell.

        Returns
        -------
        An instance of Pipeline.
        """

        self.parameters = parameters
        self.file_prefix = f"Morphomics.PID_{Parameters_ID}"
        self.morphoframe = morphoframe
        self.metadata = metadata

        self.default_params = DefaultParams()
        
        #print("Unless you have specified the file prefix in the succeeding executables, 
        print("This will be the file prefix: %s"%(self.file_prefix))
        print("")

    ## Private
    def _get_variable(self, variable_filepath, 
                       variable_name, 
                       morphoframe = True):
        """
        Finds the variable that should be processed/used by the protocol

        Parameters
        ----------
        variable_filepath (str): Path to the file that contains the variable of interest.
        variable_name (str): Name of the variable of interest in self.morphoframe.
        morphoframe (bool): Choose between self.morphoframe and self.metadata.
        
        Returns
        -------
        _morphoframe (dict): The variable used by the protocol.
        """ 

        if variable_filepath:
            print("Loading %s file..." %(variable_filepath))
            _morphoframe = morphomics.utils.load_obj(variable_filepath.replace(".pkl", ""))
        elif morphoframe:
            _morphoframe = self.morphoframe[variable_name]  
        else:
            _morphoframe = self.metadata

        return _morphoframe

    def _set_filename(self, protocol_name, save_folderpath, save_filename, default_save_filename, save_data = True):
        """
        Defines the path of the file containing the output of the protocol.

        Parameters
        ----------
        protocol_name (str): The name of the protocol.
        save_folderpath (str): The folder path containing the saved file .
        save_filename (str): Name of the file that will be saved.
        save_data (bool): Data will be saved or not.

        Returns
        -------
        save_filepath (str): The path of the file containing the output of the protocol.
        Also, update save_folderpath and save_filename of the protocol in self.parameters.
        """
        if save_data:
            if save_filename == 0:
                self.parameters[protocol_name]["save_filename"] = default_save_filename
            else:
                self.parameters[protocol_name]["save_filename"] = save_filename
            if save_folderpath == 0:
                self.parameters[protocol_name]["save_folderpath"] = os.getcwd()
            save_filepath = "%s/%s" % (self.parameters[protocol_name]["save_folderpath"], self.file_prefix + '.' + self.parameters[protocol_name]["save_filename"])
        else:
            save_filepath = None

        return save_filepath
    
    def _image_filtering(self, persistence_images, params, save_filepath):

        #if os.path.isfile(params["FilteredPixelIndex_filepath"]):
        if params["FilteredPixelIndex_filepath"]:
            print("Loading indices used for filtering persistence images...")
            _tokeep = morphomics.utils.load_obj(params["FilteredPixelIndex_filepath"].replace(".pkl", ""))
        else:
            std = str(params["pixel_std_cutoff"])
            print("Keeping pixels in persistence image with standard deviation higher than " + std + " ...")
            _tokeep = np.where(
                np.std(persistence_images, axis=0) >= params["pixel_std_cutoff"]
            )[0]
            print(len(_tokeep), 'pixels to keep over', persistence_images.shape[1])
        
        # Create a boolean mask for the rest of the elements
        mask = np.ones(len(persistence_images[0]), dtype=bool)
        mask[_tokeep] = False
        # Array with the rest of the elements
        filtered_image = np.array([np.array(pi[mask]) for pi in persistence_images])
        #filtered_image = np.array([np.array(pi[_tokeep]) for pi in persistence_images])
        print(filtered_image.shape)
        self.metadata["pixes_tokeep"] = _tokeep

        if params["save_data"]:
            print("The filtration is saved in %s" %(save_filepath))
            morphomics.utils.save_obj(self.metadata, "%s-FilteredIndex" % (save_filepath))
            morphomics.utils.save_obj(filtered_image, "%s-FilteredMatrix" % (save_filepath))
            
        return filtered_image


    ## Public
    def Input(self):
        """
        Protocol: Build panda DataFrame with cell info, load .swc files, 
            transform them into Neuron instances and store them as an element of self.morphoframe.
        
        Parameters:
        -----------
            data_location_filepath (str): Location of the parent folder containing the .swc files arranged hierarchically according to conditions.
            extension (str): .swc file extension, "_corrected.swc" refers to .swc files that were corrected with NeurolandMLConverter.
            conditions (list, str): This must match the hierarchical structure of `data_location_filepath`.
            separated_by (str): Saving chunks of the morphoframe via this condition, this must be an element of `conditions`.
            morphoframe_name (str): This is how the variable in self.morphoframe will be called.
            save_data (bool): Trigger to save output of protocol.
            save_folderpath (str): Location where to save the variable.
            save_filename (str or 0): This will be used as the file name.
        
        Returns
        -------
        Add a dataframe with key morphoframe_name to morphoframe.
        Each row in the dataframe is data from one cell and each column is a feature of the cell.
        """

        defined_params = self.parameters["Input"]
        self.default_params.check_params(defined_params, "Input")
        params = self.default_params.complete_with_default_params(defined_params, "Input")

        self.parameters["Input"] = params
        
        data_location_filepath = params["data_location_filepath"]
        extension = params["extension"]
        conditions = params["conditions"]
        separated_by = params["separated_by"]

        morphoframe_name = params["morphoframe_name"]
        
        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]        
        # define output filename
        default_save_filename = "Cell"

        print("Loading the data from %s"%(data_location_filepath))
        
        # Get DataFrame that contains context information (file_path, animal name ...) on each cell.
        info_frame = io.get_info_frame(data_location_filepath,
                                extension = extension,
                                conditions = conditions)
        # Load the data
        if separated_by is not None:
            assert (
                len(conditions) > 1
            ), "`conditions` must have more than one element. Otherwise, remove `separated_by` argument"
            assert separated_by in conditions, "`separated_by` must be in `conditions`"
            
            cond_values = info_frame[separated_by].unique()
            morphoframe = {}

            print("Separating DataFrame into %s..." % separated_by)
            print("There are %d values for the separating condition..." % len(cond_values))
            print(" ")
            for _v in cond_values:
                print("...processing %s" % _v)
                # Get the rows that respect the condition value.
                _sub_info_frame = (info_frame.loc[info_frame[separated_by] == _v]
                                    .copy()
                                    .reset_index(drop=True)
                )
                # Set the columns of swc arrays and Neuron.
                my_population = Population(info_frame = _sub_info_frame,
                                            conditions = conditions,
                                            folder_path = data_location_filepath)
                morphoframe[_v] = my_population.cells
                
                # Save the file 
                if save_data:
                    suffix = "%s-%s" % (separated_by, _v)
                    if save_filename != 0:
                        _save_filename = "%s.%s" % (save_filename, suffix)
                    else:
                        _save_filename = 0
                    _default_save_filename = "%s.%s" % (default_save_filename, suffix)
                    _save_filepath = self._set_filename(protocol_name = "Input", 
                                                            save_folderpath = save_folderpath, 
                                                            save_filename = _save_filename,
                                                            default_save_filename = _default_save_filename, 
                                                            save_data = save_data)
                    print("Saving sub dataset in %s"%(_save_filepath))
                    morphomics.utils.save_obj(morphoframe[_v], _save_filepath)
                    print("The sub dataset is saved in %s" %(_save_filepath))
                    print(" ")
            _morphoframe = pd.concat([morphoframe[_v] for _v in cond_values], ignore_index=True)
                
        else:
            # Set the columns of swc arrays and Neuron.
            my_population = Population(info_frame = info_frame,
                                        conditions = conditions,
                                        folder_path = data_location_filepath)
            _morphoframe = my_population.cells

        main_save_filepath = self._set_filename(protocol_name = "Input", 
                                            save_folderpath = save_folderpath, 
                                            save_filename = save_filename,
                                            default_save_filename = default_save_filename, 
                                            save_data = save_data)
        # Get the rows that failed to load swc array or with empty array.
        _failed_mf = _morphoframe[pd.isna(_morphoframe['cells'])]
        _failed_file_paths = list(_failed_mf['file_path'].values)

        # Get the rows that load correctly.
        _morphoframe = _morphoframe[~pd.isna(_morphoframe['cells'])]
        self.morphoframe[morphoframe_name] = _morphoframe.reset_index(drop=True)

        # save the file 
        if save_data:
            print("Saving dataset in %s"%(main_save_filepath))
            morphomics.utils.save_obj(self.morphoframe[morphoframe_name], main_save_filepath)
            print("The morphoframe is saved in %s" %(main_save_filepath))

            # Save name of failed files in .txt
            _save_failed_filepath = "%s-FailedFiles.txt" % (main_save_filepath)
            np.savetxt(_save_failed_filepath, _failed_file_paths, delimiter='\n', fmt="%s")
        
        print("...finished loading morphologies...")
        print(" ")
        nb_fails = len(_failed_file_paths)
        if nb_fails > 0:
            print(f"! Warning: You have {nb_fails} .swc files that did not load. Potentially, empty files. Please check *-FailedFiles")
    
        print("Input done!")
        print("")



    def Load_data(self):
        """
        Protocol: Load morphoframe from .pkl files.
        
        Essential parameters:
            filepath_to_data (0 or str): Full path to file to be loaded.
            morphoframe_name (str): This is how the variable in self.morphoframe will be called.
            folderpath_to_data (str): Location to the pickle file outputs to Protocols.Input.
            conditions_to_include (list of str): The different conditions that you want to load

        Returns
        -------
        Add a dataframe to morphoframe.
        """
        defined_params = self.parameters["Load_data"]
        self.default_params.check_params(defined_params, "Load_data")
        params = self.default_params.complete_with_default_params(defined_params, "Load_data")
        self.parameters["Load_data"] = params

        filepath_to_data = params["filepath_to_data"]
        morphoframe_name = params["morphoframe_name"]


        if filepath_to_data != 0:
            self.morphoframe[morphoframe_name] = load_obj(filepath_to_data)

        else:
            conditions_to_include = params["conditions_to_include"]
            folderpath_to_data = params["folderpath_to_data"]
            filename_prefix = params["filename_prefix"]
            _morphoframe = {}
            for _c in conditions_to_include:
                print("...loading %s" % _c)
                filepath = "%s/%s%s" % (folderpath_to_data, filename_prefix, _c)
                _morphoframe[_c] = morphomics.utils.load_obj(filepath.replace(".pkl", ""))

            self.morphoframe[morphoframe_name] = pd.concat([_morphoframe[_c] for _c in conditions_to_include], ignore_index=True)
        
        print('Loading done!')
        print("")



    def TMD(self):
        '''
        Protocol: Compute and Add the Topological Morphology Descriptor to morphoframe for each cell in morphoframe.

        Parameters:
        -----------
            morphoframe_filepath (str or 0): If not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name.
            filtration_function (str): This is the TMD filtration function, can either be radial_distance, or path_distance.
            exclude_sg_branches (bool): if you want to remove the branches link to the soma that do not have ramifications i.e. simple trunks.
            morphoframe_name (str): This is how the variable in self.morphoframe will be called.
            save_data (bool): Trigger to save output of protocol.
            save_folderpath (str): Location where to save the variable.
            save_filename (str or 0): This will be used as the file name.
        
        Returns
        -------
        Add the TMD (also called barcode or persistence homology) of each cell into morphoframe.
        '''
        defined_params = self.parameters["TMD"]
        self.default_params.check_params(defined_params, "TMD")
        params = self.default_params.complete_with_default_params(defined_params, "TMD")
        self.parameters["TMD"] = params

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]

        filtration_function = params["filtration_function"]
        exclude_sg_branches = params["exclude_sg_branches"]
        
        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        # initialize morphoframe to bootstrap
        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                            variable_name = morphoframe_name)   

        cells = _morphoframe.copy()
        my_population = Population(cells_frame = cells)
        
        if exclude_sg_branches:
                my_population.exclude_sg_branches()

        print("Computing the TMD on morphoframe %s"%(morphoframe_name))
        my_population.set_barcodes(filtration_function = filtration_function)
        _morphoframe = my_population.cells
        _morphoframe = _morphoframe[~pd.isna(_morphoframe['barcodes'])]

        # define output filename
        default_save_filename = "TMD"
        save_filepath = self._set_filename(protocol_name = "TMD", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)
        
        self.morphoframe[morphoframe_name] = _morphoframe

        # save the file 
        if save_data:
            print("Saving dataset in %s"%(save_filepath))
            morphomics.utils.save_obj(self.morphoframe[morphoframe_name], save_filepath)
            print("The TMD morphoframe is saved in %s" %(save_filepath))

        print("...finished computing barcodes...")
        print("TMD done!")
        print("")



    def Clean_frame(self):
        """
        Protocol: Clean out the morphoframe, to filter out artifacts and unwanted conditions.
        
        Essential parameters:
            morphoframe_filepath (str or 0): If not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name.
            morphoframe_name (str): Key of the morphoframe which will be filtered out.
            barcode_size_cutoff (int): Remove morphologies if the number of bars is less than the cutoff
            barlength_cutoff (list, (str, float)): Retain bars whose length satisfy a certain cutoff
                                    must be an array with two elements, [">" "<", ">=", "<=", "==", bar length cutoff].
            combine_conditions (list, (str, list, str)): # enumerate which conditions will be merged
                                    must be an array with three elements 
                                        [a header of the info_frame (is an element of `Input.conditions`),
                                         a list of conditions that will be merged (must be an array), 
                                        the new name of the merged conditions]
            restrict_conditions (list, (str, list, str)): enumerate restrictions
                                        must be an array with three elements:
                                            [a header of the info_frame (is an element of `Input.conditions`),  
                                            list of conditions to either drop or keep (must be an array), 
                                            "drop" or "keep" conditions specified]
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix

        Returns
        -------
        Add a dataframe to morphoframe. the samples that don't respond to the conditions are removed.
        """
        defined_params = self.parameters["Clean_frame"]
        #self.default_params.check_params(defined_params, "Clean_frame")
        params = self.default_params.complete_with_default_params(defined_params, "Clean_frame")
        self.parameters["Clean_frame"] = params

        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        # define morphoframe to clean
        _morphoframe = self._get_variable(variable_filepath = params["morphoframe_filepath"],
                                            variable_name = params["morphoframe_name"])

        # drops empty morphologies, potentially artifacts
        _morphoframe = _morphoframe.loc[~_morphoframe.barcodes.isna()].reset_index(
            drop=True
        )

        # barcode size filtering
        barcode_size_cutoff = float(params["barcode_size_cutoff"])
        print("Removing morphologies with barcode size less than %.2f..."%barcode_size_cutoff)
        _morphoframe["Barcode_length"] = _morphoframe.barcodes.apply(lambda x: len(x))
        _morphoframe = _morphoframe.query(
            "Barcode_length >= @barcode_size_cutoff"
            ).reset_index(drop=True)

        # bar length filtering
        for _operation, barlength_cutoff in params["barlength_cutoff"]:
            print("Removing bars from all barcodes with the following criteria: bar length %s %.2f"%(_operation, float(barlength_cutoff)))
            _morphoframe.barcodes = _morphoframe.barcodes.apply(lambda x: morphomics.Topology.analysis.filter_ph(x, float(barlength_cutoff), method=_operation))
            
        # replace/rename/combine conditions
        if len(params["combine_conditions"]) > 0:
            for _cond, _before, _after in params["combine_conditions"]:
                print("Replacing all instances of `%s` in the `%s` morphoframe column with %s"%(_before, _cond, _after))
                _morphoframe.loc[_morphoframe[_cond].isin(_before), _cond] = _after

        # restrict conditions
        if len(params["restrict_conditions"]) > 0:
            for _cond, _restricts, _action in params["restrict_conditions"]:
                assert _cond in _morphoframe.keys(), "%s not in morphoframe..."%_cond
                if _action == "drop":
                    for _restrictions in _restricts:
                        print("Filtering out %s from %s..."%(_restrictions, _cond))
                        assert _restrictions in _morphoframe[_cond].unique(), "%s not in the available condition..."%_restrictions
                        _morphoframe = _morphoframe.loc[~_morphoframe[_cond].str.contains(_restrictions)].reset_index(drop=True)
                elif _action == "keep":
                    _restrictions = "|".join(_restricts)
                    print("Keeping %s in %s..."%(_restrictions, _cond))
                    _morphoframe = _morphoframe.loc[_morphoframe[_cond].str.contains(_restrictions)].reset_index(drop=True)
                else:
                    print("Warning for ", _cond, _restricts)
                    print("Third column must be either 'drop' or 'keep'...")
                    print("Nothing will be done...")

        print("Clean done!")
        
        # initialize output filename
        default_save_filename = "Cleaned"
        save_filepath = self._set_filename(protocol_name = "Clean_frame", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)
        
        # change file name prefix
        # save the file 
        if params["save_data"]:
            morphomics.utils.save_obj(_morphoframe, save_filepath)
            print("The cleaned morphoframe is saved in %s" %(save_filepath))

        self.morphoframe[params["morphoframe_name"]] = _morphoframe

     
     
    def Subsample(self):
        """
        Protocol: subsample/modify a tree or a barcode to reduce its "noise".

        Essential parameters:
            morphoframe_filepath (str or 0): If not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name.
            morphoframe_name (str): Key of the morphoframe which will be subsampled out.
            feature_to_subsample (str): A column in morphoframe .i.e the type (either barcodes, or tree) to subsample.
            main_branches (str or None): If you want to force main branches to be kept = 'keep'. 
                                        If you want to remove them, and keep only subbranches = 'remove'.
            k_elements (int or ratio): The number of elements that will be subsampled to generate a subbarcode or subtree.
            n_samples (int): Number of subbarcodes per barcode.
            rand_seed (int): Seed of the random number generator.
            extendedframe_name (str): Where the subsampled morphoframe will be stored.
            save_data (bool): Trigger to save output of protocol.
            save_folderpath (str): Location where to save the data.
            save_filename (str or 0): Name of the file containing the bootstrap frame.

        """
        defined_params = self.parameters["Subsample"]
        self.default_params.check_params(defined_params, "Subsample")
        params = self.default_params.complete_with_default_params(defined_params, "Subsample")
        self.parameters["Subsample"] = params

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]
        
        extendedframe_name = params["extendedframe_name"]

        feature_to_subsample = params["feature_to_subsample"]
        #could be a ratio of the number of bars
        n_samples = params["n_samples"]

        rand_seed = params["rand_seed"]

        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        # initialize morphoframe to bootstrap
        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                            variable_name = morphoframe_name)   
        _morphoframe_copy = _morphoframe.copy()

        features = _morphoframe_copy[feature_to_subsample]
        if feature_to_subsample == "barcodes":
            main_branches = params["main_branches"]
            k_elements = params["k_elements"]
            _morphoframe_copy[feature_to_subsample + "_proba"] = subsampler.set_proba(#feature_to_subsample = feature_to_subsample,
                                                                                        feature_list = features, 
                                                                                        main_branches = main_branches)
            probas = _morphoframe_copy[feature_to_subsample + "_proba"]
            _morphoframe_copy[feature_to_subsample + "_subsampled"] = subsampler.subsample_w_replacement(#feature_to_subsample = feature_to_subsample,
                                                                                                        feature_list = features,
                                                                                                        probas = probas, 
                                                                                                        k_elements = k_elements, 
                                                                                                        n_samples = n_samples, 
                                                                                                        rand_seed = rand_seed,
                                                                                                        main_branches = main_branches)
        else:
            _type = params['type']
            number = params['nb_sections']
            _morphoframe_copy[feature_to_subsample + "_subsampled"] = subsampler.subsample_trees(feature_list = features,
                                                                                                type = _type,
                                                                                                number = number,
                                                                                                n_samples = n_samples, 
                                                                                                rand_seed = rand_seed,)

        _morphoframe_copy[feature_to_subsample + '_id'] = _morphoframe_copy.index
        extendedframe = _morphoframe_copy.explode(feature_to_subsample + "_subsampled").reset_index(drop = True)

        # initialize output filename
        default_save_filename = "Subsampled"
        save_filepath = self._set_filename(protocol_name = "Subsample", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)
        
        self.morphoframe[morphoframe_name] = _morphoframe_copy
        self.morphoframe[extendedframe_name] = extendedframe[[feature_to_subsample + "_subsampled", feature_to_subsample + '_id']]

        # save the file 
        if params["save_data"]:
            morphomics.utils.save_obj(self.morphoframe[morphoframe_name], save_filepath)
            morphomics.utils.save_obj(self.morphoframe[extendedframe_name], save_filepath + '_extended')
            print("The Subsampled morphoframe is saved in %s" %(save_filepath))

        print("Subsampling done!")
        print("")



    def Bootstrap(self):
        """
        Protocol: Takes the morphoframe and bootstraps the variable specified in  `feature_to_bootstrap` and returns a new morphoframe with the bootstrapped samples
        
        Essential parameters:
            morphoframe_filepath (str or 0): If not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name.
            morphoframe_name (str): Key of the morphoframe which will be bootstrapped out.
            feature_to_bootstrap (list, (str, str)): A column in morphoframe and the type (either bars, scalar or array) to bootstrap.
            bootstrap_conditions (list, str): Conditions to bootstrap together. If you want to bootstrap over all the conditions in "morphoframe_name", then leave the "bootstrap_conditions" empty
            rand_seed (int): Seed of the random number generator.
            N_bags (int): Number of bootstrapped/averaged points in one population (i.e. per combination of conditions).
            n_samples (int): Number of sampled points to create a bootstrapped point.
            ratio (float): Only used if n_samples = 0. A number between 0 and 1, defines the number of samples per population with respect to the pop size. 
            bootstrapframe_name (str): Where the bootstrapped morphoframe will be stored.
            save_data (bool): Trigger to save output of protocol.
            save_folderpath (str): Location where to save the data.
            save_filename (str or 0): Name of the file containing the bootstrap frame.

        Returns
        -------
        Add a dataframe containing bootstrapped data to morphoframe. The samples are bootstrapped points of microglia.
        """
        defined_params = self.parameters["Bootstrap"]
        self.default_params.check_params(defined_params, "Bootstrap")
        params = self.default_params.complete_with_default_params(defined_params, "Bootstrap")
        self.parameters["Bootstrap"] = params

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]

        feature_to_bootstrap = params["feature_to_bootstrap"]
        bootstrap_conditions = params["bootstrap_conditions"]

        N_bags = params["N_bags"]
        n_samples = params["n_samples"]
        ratio = params["ratio"]

        rand_seed = params["rand_seed"]

        bootstrapframe_name = params["bootstrapframe_name"]
        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]
                
        # initialize morphoframe to bootstrap
        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                            variable_name = morphoframe_name)   
        _morphoframe_copy = _morphoframe.copy()

        print("Bootstrapping with the following parameters: ")
        print("bootstrap resolution: %s"%("-".join(bootstrap_conditions)))
        if n_samples == 0 :
            print("bag size = n_samples/pop_size: %d"%(int(ratio)))
        else:
            print("bag size: %d"%(int(n_samples)))
        print("number of bootstrap bags: %d"%(int(N_bags)))
        bootstrapped_frame = (
            morphomics.bootstrapping.get_bootstrap_frame(
                _morphoframe_copy,
                feature_to_bootstrap = feature_to_bootstrap,
                bootstrap_conditions = bootstrap_conditions,
                N_bags = N_bags,
                replacement = True,
                n_samples = n_samples,
                ratio = ratio,
                rand_seed = rand_seed,
            )
        )

        # define output filename
        default_save_filename = "Bootstrapped"
        save_filepath = self._set_filename(protocol_name = "Bootstrap", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)
        
        self.morphoframe[bootstrapframe_name] = bootstrapped_frame
        
        if save_filepath is not None:
            save_obj(self.morphoframe[bootstrapframe_name], save_filepath)
            print("The bootstraped morphoframe is saved in %s" %(save_filepath))

        print("Bootstrap done!")
        print("")



    def Vectorizations(self):
        """
        Protocol: Takes a morphoframe called morphoframe_name and vectorize the barcodes within the morphoframe.
        Must be a column called barcodes.
        
        Essential parameters:
            morphoframe_filepath (str or 0): If not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name.
            morphoframe_name (str): Key of the morphoframe which will be vectotized out.
            vect_method_parameters (dict): Keys are the vectorization methods applied on the TMD and attributes are the parameters of each vetorization method
            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            save_filename (str or 0): this will be used as the file name

        Returns
        -------
        Add a list of vectors to a morphoframe. A row per sample, the colums are the dimensions of the vector (result of the TMD vectorization).
        """

        defined_params = self.parameters["Vectorizations"]
        self.default_params.check_params(defined_params, "Vectorizations")
        params = self.default_params.complete_with_default_params(defined_params, "Vectorizations")
        self.parameters["Vectorizations"] = params

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]
        vect_method_parameters = params["vect_method_parameters"]
        
        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        # define morphoframe containing barcodes to compute vectorizations
        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                           variable_name = morphoframe_name)
        _morphoframe_copy = _morphoframe.copy()
        assert (
            "barcodes" in _morphoframe_copy.keys()
        ), "Missing `barcodes` column in info_frame..."

        # define the name of the vect method
        vect_methods = vect_method_parameters.keys()
        vect_methods_names = [vectorization_codenames[vect_method] for vect_method in vect_methods]
        vect_methods_codename = '_'.join(vect_methods_names)

        if '_' not in vect_methods_codename:
            print("Computes %s." %(vect_methods_codename))
        else:
            print("Computes %s and concatenates the vectors from the same microglia." %(vect_methods_codename))
        
        # initalize an instance of Vectorizer
        vectorizer = Vectorizer(tmd = _morphoframe_copy["barcodes"], 
                                vect_parameters = vect_method_parameters)
        
        # compute vectors
        output_vectors = []
        for vect_method in vect_methods:
            perform_vect_method = getattr(vectorizer, vect_method)
            output_vector = perform_vect_method()
            
            output_vectors.append(output_vector)
        output_vectors = np.concatenate(output_vectors, axis=1)

        # define output filename
        default_save_filename = "Vectorizations-%s"%(vect_methods_codename)
        save_filepath = self._set_filename(protocol_name = "Vectorizations", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)
        
        self.morphoframe[morphoframe_name][vect_methods_codename] = list(output_vectors)

        print("Vectorization done!")

        # save the output vectors
        if save_filepath is not None:
            print("The vectors are saved in %s" %(save_filepath))
            save_obj(obj = self.morphoframe[morphoframe_name], filepath = save_filepath)



    def Dim_reductions(self):
        """
        Protocol: Takes the vectors you want in morphoframe and reduces them following the chosen dim reduction techniques
              
        Essential parameters:
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be processed out
            dimred_method_parameters (dict): keys are the reducer names applied to the vectors and attributes are the parameters of each reducer
            vectors_to_reduce (str): the name of the vectors to reduce in morphoframe
            filter_pixels (bool): filter persistence image
            FilteredPixelIndex_filepath (float): spread of the Gaussian kernel
            pixel_std_cutoff (str): how to normalize the persistence image, can be "sum" or "max"
            normalize (bool): normalize data before reduction
            standardize (bool): standardize data before reduction
            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            save_filename (str or 0): This will be used as the file name.

        Returns
        -------
        Add a list of reduced vectors to a morphoframe. A row per sample (example: microglia), the colums are the dimensions of the reduced vectors (result of the dimensionality reduction).
        """
        defined_params = self.parameters["Dim_reductions"]
        self.default_params.check_params(defined_params, "Dim_reductions")
        params = self.default_params.complete_with_default_params(defined_params, "Dim_reductions")
        self.parameters["Dim_reductions"] = params

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]

        dimred_method_parameters = params["dimred_method_parameters"]
        vectors_to_reduce = params["vectors_to_reduce"]
        filter_pixels = params["filter_pixels"]
        normalize = params['normalize']
        standardize = params['standardize']

        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]
        save_dimreducer = params["save_dimreducer"]

        # define morphoframe containing vectors to compute dim reduction
        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                           variable_name = morphoframe_name)
        _morphoframe_copy = _morphoframe.copy()

        X = np.vstack(_morphoframe_copy[vectors_to_reduce])
        
        dimred_methods = dimred_method_parameters.keys()
        dimred_method_names = '_'.join(list(dimred_methods))
        # define output filename
        default_save_filename = "DimReductions-%s"%(dimred_method_names)
        save_filepath = self._set_filename(protocol_name = "Dim_reductions", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)
        # if persistence image, pixels can be filtered 
        if filter_pixels:
            filtered_image = self._image_filtering(persistence_images = X,
                                                  params = params, 
                                                  save_filepath = save_filepath)
            X = filtered_image

        # normalize data 
        if normalize:
            print("Normalize the vectors")
            normalizer = Normalizer()
            X = normalizer.fit_transform(X)
        # standardize data 
        if standardize:
            print("Standardize the vectors")
            standardize = StandardScaler()
            X = standardize.fit_transform(X)

        print("Reduces the vectors with the following techniques %s " %(dimred_method_names))
        # initialize an instance of DimReducer
        dimreducer = DimReducer(tmd_vectors = X,
                                dimred_parameters = dimred_method_parameters)
        
        # dim reduce the vectors
        fit_dimreducers = []
        for dimred_method in dimred_methods:
            perform_dimred_method = getattr(dimreducer, dimred_method)
            fit_dimreducer, reduced_vectors = perform_dimred_method()

            fit_dimreducers.append(fit_dimreducer)
   
            dimreducer.tmd_vectors = reduced_vectors

        self.metadata['fitted_' + dimred_method_names] = fit_dimreducers

        self.morphoframe[morphoframe_name][dimred_method_names] = list(reduced_vectors)

        # save the reduced vectors
        if save_filepath is not None:
            save_obj(obj = self.morphoframe[morphoframe_name], filepath = save_filepath + '_reduced_data')
            print("The reduced vectors are saved in %s" %(save_filepath))
        if save_dimreducer:
            save_obj(obj = self.metadata, filepath = save_filepath + '_fitted_dimreducer')
            print("The fitted dimreducers are saved in %s" %(save_filepath))

        print("Reducing done!")
        print("")



    def Palantir(self):
        #     """
        #     Protocol: Takes the UMAP manifold, calculates diffusion maps using Palantir and outputs a force-directed layout of the maps
            
        #     Essential parameters:
        #         X_umap_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
        #         n_diffusion_components (int): number of diffusion maps to generate
        #         knn_diffusion (int): number of nearest neighbors that will be used to generate diffusion maps
        #         fdl_random_seed (float): seed of the random number generator for the force-directed layout
        #         save_data (bool): trigger to save output of protocol
        #         save_folder (str): location where to save the data
        #         file_prefix (str or 0): this will be used as the file prefix
        #     """
        #     params = self.parameters["Palantir"]
            
        #     # define output filename
        #     file_prefix = "%s.Palantir"%(self.file_prefix)
        #     save_filename = self._set_filename(protocol_name = "Palantir", 
        #                                           save_folder_path = params["save_folder"], 
        #                                           file_prefix = file_prefix, 
        #                                           save_data = params["save_data"])
                
        #     if params["X_umap_filepath"]:
        #         print("Loading UMAP coordinates...")
        #         self.metadata["X_umap"] = morphomics.utils.load_obj(params["X_umap_filepath"].replace(".pkl", ""))
            
        #     print("Calculating diffusion maps with Palantir...")
        #     self.metadata["palantir_distances"] = morphomics.old_reduction.palantir_diffusion_maps(
        #         self.metadata["X_umap"], 
        #         n_components=params["n_diffusion_components"], 
        #         knn=params["knn_diffusion"],
        #     )

        #     print("Calculating 2D coordinates with force-directed layout")
        #     self.metadata["X_fdl"] = morphomics.old_reduction.force_directed_layout(
        #         self.metadata["palantir_distances"]["kernel"], 
        #         random_seed=params["fdl_random_seed"]
        #     )

        #     print("Palantir done!")
            
        #     if params["save_data"]:
        #         print("The palantir is saved in %s" %(save_filepath))
        #         morphomics.utils.save_obj(self.metadata["palantir_distances"], "%s-PalantirDistances" % (save_filename) )
        #         morphomics.utils.save_obj(self.metadata["X_fdl"], "%s-PalantirFDCoords" % (save_filename) )
        return


            
    def Save_reduced(self):
        """
        Protocol: Takes the reduced manifold coordinates and conditions to create a .csv file which can be uploaded to the morphOMICs dashboard
        
        Parameters:
        -----------
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be processed out
            conditions_to_save (list of str): the list of the keys in morphoframe you want to save with reduced vectors
            dimred_method (str): name of the colum containing reduced vectors to save
            coordinate_axisnames(str): general name of the saved columns in csv 
            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            save_filename (str or 0): this will be used as the file name

        Returns
        -------
        A saved .csv file containing a column for each wanted condition and a column for each dimension of the reduced vectors. 
        A row per sample.
        """
        defined_params = self.parameters["Save_reduced"]
        self.default_params.check_params(defined_params, "Save_reduced")
        params = self.default_params.complete_with_default_params(defined_params, "Save_reduced")
        self.parameters["Save_reduced"] = params

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]

        conditions_to_save = params["conditions_to_save"]
        dimred_method = params["dimred_method"]
        coordinate_axisnames = params["coordinate_axisnames"]

        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        print("Preparing .csv file that can be loaded into the morphOMICs dashboard...")

        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath, 
                                           variable_name = morphoframe_name)
        
        _submorphoframe_copy = _morphoframe[conditions_to_save].copy()
        reduced_vectors = _morphoframe[dimred_method].copy()
        reduced_vectors = np.vstack(reduced_vectors)

        for dims in range(reduced_vectors.shape[1]):
            _submorphoframe_copy["%s_%d"%(coordinate_axisnames, dims+1)] = reduced_vectors[:, dims]
        

        # define output filename
        default_save_filename = "ReductionInfo"
        save_filepath = self._set_filename(protocol_name = "Save_reduced", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename)
        print(save_filepath)
        save_filepath = "%s.csv" % (save_filepath)

        # ensure the directory exists
        save_directory = os.path.dirname(save_filepath)
        os.makedirs(save_directory, exist_ok = True)
        # save the csv file
        _submorphoframe_copy.to_csv(save_filepath, index = True)

        print("Reduced coordinates splitted and saved!")
        print("")



    def Mapping(self):
        """
        Protocol: Takes a pre-calculated UMAP function, maps persistence images into the UMAP manifold and outputs the manifold coordinates
        
        Essential parameters:
            fitted_dimreducer_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            dimred_method (str): the dim reducer fitted and used for dim reduction.
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be processed out
            vectors_to_reduce_name (str): name of the vectors in morphoframe that will be reduced 
            filter_pixels (1 or 0): prompt to filter pixels in the persistence images
            FilteredPixelIndex_filepath (str): location of the filtered pixel indices before doing the UMAP of the generated phenotypic spectrum
            pixel_std_cutoff (str): how to normalize the persistence image, can be "sum" or "max"
            normalize (bool): normalize data or not 
            standardize (bool): standardize data or not 

            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            save_filename (str or 0): this will be used as the file name

        Returns
        -------
        Add a list of reduced vectors to a morphoframe. A row per sample (example: microglia), the colums are the dimensions of the reduced vectors (result of the dimensionality reduction).
        """
        defined_params = self.parameters["Mapping"]
        self.default_params.check_params(defined_params, "Mapping")
        params = self.default_params.complete_with_default_params(defined_params, "Mapping")
        self.parameters["Mapping"] = params

        fitted_dimreducer_filepath = params["fitted_dimreducer_filepath"]
        dimred_method = params["dimred_method"]

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]
        vectors_to_reduce_name = params["vectors_to_reduce_name"]

        filter_pixels = params["filter_pixels"]
        FilteredPixelIndex_filepath = params["FilteredPixelIndex_filepath"]
        
        normalize = params['normalize']
        standardize = params['standardize']

        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        # define morphoframe that contains the fitted reducer
        print("Loading fitted dim reduction function...")    
        _metadata = self._get_variable(variable_filepath = fitted_dimreducer_filepath,
                                            variable_name = None,
                                            morphoframe = False)
        f_dimreducers = _metadata[dimred_method]

        print("Loading persistence vectors to reduce file...")
        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                            variable_name = morphoframe_name)
        
        vectors_to_reduce = _morphoframe[vectors_to_reduce_name].copy()
        vectors_to_reduce = np.vstack(vectors_to_reduce)
       
        # define output filename
        default_save_filename = "Mapping"
        save_filepath = self._set_filename(protocol_name = "Mapping", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)     

        if filter_pixels and os.path.isfile(FilteredPixelIndex_filepath):
            self._image_filtering(persistence_images = vectors_to_reduce,
                                    params = params, 
                                    save_filename = save_filepath)
        if normalize:
            print("Normalize the vectors")
            normalizer = Normalizer()
            vectors_to_reduce = normalizer.fit_transform(vectors_to_reduce)
        if standardize:
            print("Standardize the vectors")
            standardizer = StandardScaler()
            vectors_to_reduce = standardizer.fit_transform(vectors_to_reduce)

        print("Mapping vectors into the reduced space...")
        for f_dimreducer in f_dimreducers:
            reduced_vectors = f_dimreducer.transform(vectors_to_reduce)
            vectors_to_reduce = reduced_vectors.copy()

        self.morphoframe[morphoframe_name] = _morphoframe
        self.morphoframe[morphoframe_name][dimred_method + '_transformed'] = list(reduced_vectors)

        if save_data:
            print("The reduced vectors are saved in %s" %(save_filepath))

            morphomics.utils.save_obj(obj = self.morphoframe[morphoframe_name],
                                      filepath = "%s-reduceCoords%dD" % (save_filepath, reduced_vectors.shape[1]) )
        
        print("Mapping done!")
        print("")

       
        
    def Sholl_curves(self):
        """
        Protocol: Takes .swc files, reads them and calculates Sholl curves at given radial intervals
        
        Essential parameters:
            morphoframe_name (str): morphoframe which contains Filenames of morphologies for Sholl analysis
            Empty_indicator (str): column name in morphoframe which will be the indicator for empty morphologies
            swc_types (str or 0): if not 0, must be a comma-separated string of numbers corresponding to swc TypeID to be considered, e.g. "2,3,4,5"
            Sholl_radius (float): radial difference between the concentric spheres
            Sholl_colname (str): key to the metadata where the Sholl curves will be stored
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Sholl_curves"]
            
        # define output filename
        file_prefix = "%s.Sholl"%(self.file_prefix)
        save_filename = self._set_filename(protocol_name = "Sholl_curves", 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])   
            
        sholl_plots = []
        
        assert params["morphoframe_name"] in self.morphoframe.keys(), "There is no `morphoframe_name`. Check this or make sure that you ran either `Input` or `Load_data` first."
        assert "Filenames" in self.morphoframe[params["morphoframe_name"]].columns, "There is no Filename column in the `morphoframe`. Make sure that you ran either `Input` or `Load_data` properly."
        assert params["Empty_indicator"] in self.morphoframe[params["morphoframe_name"]].columns, "There is no column with the assigned `Empty_indicator` in the `morphoframe`. Check the morphoframe structre."
        
        files = self.morphoframe[params["morphoframe_name"]].Filenames
        empty_morphologies = self.morphoframe[params["morphoframe_name"]][params["Empty_indicator"]].isna()

        print("Calculating Sholl curves with radius %.2f"%(float(params["Sholl_radius"])))
        
        for _idx, (filename, empty) in enumerate(zip(files, empty_morphologies)):
            if _idx%10==0: 
                print("Sholl curve calculation is now at %s"%(filename.split("/")[-1]))
            
            if not empty:
                s = morphomics.morphometrics.calculate_sholl_curves(filename, params["Sholl_radius"], _type=params["swc_types"])
            else:
                s = []
            sholl_plots.append(s)
            
        sholl_curves = pd.DataFrame(sholl_plots, columns=["Sholl_curves"])
        sholl_curves["Files"] = files
        self.metadata[params["Sholl_colname"]] = sholl_curves[["Files", "Sholl_curves"]]
        
        print("Sholl done!")
        print("")

        if params["save_data"]:
            #print("The Sholl curves are saved in %s" (save_filepath))
            morphomics.utils.save_obj(self.metadata, "%s" % (save_filename) )

            
            
    def Morphometrics(self):
        """
        Protocol: Takes .swc files, reads them and calculates classical morphometric quantities
        
        Essential parameters:
            morphoframe_name (str): morphoframe which contains Filenames of morphologies for Sholl analysis
            Empty_indicator (str): column name in morphoframe which will be the indicator for empty morphologies
            temp_folder (str): location where to store .swc files that contain spaces in their filename
            Lmeasure_functions (list, (float, float)): list containing morphometric quantities of interests, must be (Lmeasure function, "TotalSum", "Maximum", "Minimum", "Average")
            Morphometric_colname (str): key to the metadata where the morphometrics will be stored
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Morphometrics"]

        # define output filename
        file_prefix = "%s.Morphometrics"%(self.file_prefix)
        save_filename = self._set_filename(protocol_name = "Morphometrics", 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])   
            
        assert params["morphoframe_name"] in self.morphoframe.keys(), "There is no `morphoframe_name`. Check this or make sure that you ran either `Input` or `Load_data` first."
        assert "Filenames" in self.morphoframe[params["morphoframe_name"]].columns, "There is no Filename column in the `morphoframe`. Make sure that you ran either `Input` or `Load_data` properly."
        assert params["Empty_indicator"] in self.morphoframe[params["morphoframe_name"]].columns, "There is no column with the assigned `Empty_indicator` in the `morphoframe`. Check the morphoframe structre."
        
        print("Calculating classical morphometric quantities...")
        
        Lm_functions, Lm_quantities = morphomics.morphometrics.create_Lm_functions(params["Lmeasure_functions"])
        
        files = self.morphoframe[params["morphoframe_name"]].Filenames
        empty_morphologies = self.morphoframe[params["morphoframe_name"]][params["Empty_indicator"]].isna()
        
        non_empty_files = files[~empty_morphologies]
        non_empty_files, morphometric_quantities = morphomics.morphometrics.calculate_morphometrics(
            non_empty_files, params["temp_folder"], Lm_functions, Lm_quantities)
        
        morphometrics = pd.DataFrame(
            morphometric_quantities,
            columns=[
                "%s_%s" % (func_i[0], func_i[1]) for func_i in params["Lmeasure_functions"]
            ],
        )
        morphometrics["Files"] = non_empty_files
        morphometrics = morphometrics[np.hstack(["Files", morphometrics.columns[:-1]])]
        self.metadata[params["Morphometric_colname"]] = morphometrics
        
        print("Morphometrics done!")
        print("")

        if params["save_data"]:
            #print("The Sholl curves are saved in %s" (save_filepath))

            morphomics.utils.save_obj(self.metadata, "%s" % (save_filename) )
             


    def Plotting(self):
        """
        Protocol: Generates a 3D interactive plot from a morphoframe, or from the ReductionInfo files, or from coordinate and morphoinfo files
        
        Essential parameters:
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): 
            conditions (list (str)): Conditions that will be concatenated to represent labels in the plot.
            reduced_vectors_name (str): The name of the column in morphoframe that contains the vectors (dim 2/3) you want to plot.
            axis_labels (list (str)): The name of the axis you want to plot.
            title (str): The title of the plot.
            colors (dict or list): The dictionnary or the list for the colors for different lables. If empty list, the colors are choosen by default.
            size (float): Size of the markers.
            amount (float [0,1]): If using different shades of the same color, this is a factor for the shades.
            save_data (bool): Trigger to save output of protocol.
            save_folderpath (str): Location where to save the data.
            save_filename (str): This will be used as the file name.
        """
        defined_params = self.parameters["Plotting"]
        self.default_params.check_params(defined_params, "Plotting")

        params = self.default_params.complete_with_default_params(defined_params, "Plotting")
        self.parameters["Plotting"] = params

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]
        
        conditions = params['conditions']
        reduced_vectors_name = params["reduced_vectors_name"]
        axis_labels = params['axis_labels']
        title = params['title']
        colors = params['colors']
        circle_color = params['circle_colors']
        size = params['size']
        amount = params['amount']

        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"] 

        # define morphoframe that contains the data points to plot
        print("Loading fitted dim reduced data...")   

        if type(morphoframe_filepath) is str:
            if "csv" in morphoframe_filepath:
                _morphoframe = pd.read_csv(morphoframe_filepath, index_col = 0)
        else:
            _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                                variable_name = morphoframe_name)
            # one column per coordinate
            _morphoframe = _morphoframe.copy()
            reduced_vectors = _morphoframe[reduced_vectors_name].copy()
            reduced_vectors = np.vstack(reduced_vectors)
            nb_dims = reduced_vectors.shape[1]
            for dims in range(nb_dims):
                _morphoframe[axis_labels[dims]]  = reduced_vectors[:, dims]

        if nb_dims >= 3:
            fig3d = plotting.plot_3d_scatter(morphoframe = _morphoframe,
                                    axis_labels = axis_labels,
                                    conditions = conditions,
                                    colors = colors,
                                    circle_color = circle_color,
                                    amount= amount,
                                    size = size,
                                    title = title)
        if nb_dims >= 2:
            fig2d = plotting.plot_2d_scatter(morphoframe = _morphoframe,
                                    axis_labels = axis_labels,
                                    conditions = conditions,
                                    colors = colors,
                                    circle_color = circle_color,
                                    amount= amount,
                                    size = size,
                                    title = title)
        
        # define output filename
        default_save_filename = "Plotting"
        save_filepath = self._set_filename(protocol_name = "Plotting", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename)

        if save_data:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
            # Save the plot as an HTML file
            if nb_dims >= 3:
                fig3d.write_html(save_filepath + '3d.html')
                self.metadata['fig3d'] = fig3d

            #fig3d.write_image(save_filepath + '3d.pdf', format = 'pdf')
            #fig2d.write_html(save_filepath + '2d.html')
            if nb_dims >= 2:
                fig2d.write_image(save_filepath + '2d.pdf', format = 'pdf')
                self.metadata['fig2d'] = fig2d
            
            #save_obj(obj = self.metadata, filepath = save_filepath + '_figures')
            print(f"Plot saved as {save_filepath}")
        print("Plotting done!")
        print("")



    def Save_parameters(self):
        """
        Protocol: Save the wanted parameters in a .pkl for reproducibility.

        Parameters:
        -----------
            parameters_to_save (dict of lists): The names of the parameters for wish you want to store the values.
                                        Should be of the form: {protocol_name_1 : [param_name_1_1, ..., param_name_N_M]
                                                                ...
                                                                protocol_name_N : [param_name_N_1, ..., param_name_N_K]
                                                                }
            save_folderpath (str): The path to the folder where the file containing parameters will be saved.
            save_filename (str): Name of the saved .pkl file.

        Returns:
        --------
            The .pkl file containg a dict of the parameters and their values.
        """
        # Save a dictionary containing the name of the processed data and the parameters of the main steps for reproducibility
        defined_params = self.parameters['Save_parameters']
        params = self.default_params.complete_with_default_params(defined_params, "Save_parameters")
        self.parameters["Save_parameters"] = params

        parameters_to_save = params['parameters_to_save']
        stored_parameters = {}
        for protocol_name in parameters_to_save.keys():
            stored_parameters[protocol_name] = {}
            for param_name in parameters_to_save[protocol_name]:
                stored_parameters[protocol_name][param_name] = self.parameters[protocol_name][param_name]

        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        default_save_filename = "Experiment_Parameters"
        save_filepath = self._set_filename(protocol_name = "Save_parameters", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename)
        
        self.metadata['exp_param'] = stored_parameters
        morphomics.utils.save_obj(obj = stored_parameters,
                                    filepath = save_filepath) 
        print("The experiment parameters are saved in %s" %(save_filepath))
        print("")



    def Clear_morphoframe(self):
        """
        Protocol: Clears morphoframe
        """
        print("Clearing morphoframe...")
        self.morphoframe = {}