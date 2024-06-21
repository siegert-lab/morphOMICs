import morphomics
import morphomics.io

import os

from morphomics.Analysis.vectorizer import Vectorizer
from morphomics.Analysis.dim_reducer import DimReducer
from morphomics.Analysis import plotting

from morphomics.utils import save_obj, load_obj, vectorization_codenames
from sklearn.preprocessing import Normalizer, StandardScaler

import numpy as np
import pandas as pd
import ipyvolume as ipv  # https://ipyvolume.readthedocs.io/en/latest/install.html
from matplotlib import cm, colors
import matplotlib.pyplot as plt

class Protocols(object):
    
    def __init__(self, parameters, Parameters_ID, morphoframe = {}, metadata = {}) :
        """
        Initializes the Protocols instance.

        Parameters
        ----------
        parameters (dict): contains the parameters for each protocol that would be run
                            parameters = {'protocol_1 : { parameter_1_1: x_1_1, ..., parameter_1_n: x_1_n},
                                            ...
                                            'protocol_m : { parameter_m_1: x_m_1, ..., parameter_m_n: x_m_n}
                                        } 
        Parameters_ID (str): a way to characterize the name of all saved files from the same Protocols instance i.e. prefix
        morphoframe (dict): contais the data from cells, each column is a feature and each row is a cell
        metadata (dict): contains the tools that the pipeline use, contains the data that are not ordered by row/cell

        Returns
        -------
        An instance of protocols.
        """

        self.parameters = parameters
        self.file_prefix = f"Morphomics.PID_{Parameters_ID}"
        self.morphoframe = morphoframe
        self.metadata = metadata
        
        #print("Unless you have specified the file prefix in the succeeding executables, 
        print("this will be the file prefix: %s"%(self.file_prefix))

    ## Private
    def _get_variable(self, variable_filepath, 
                       variable_name, 
                       morphoframe = True):
        """
        Finds the variable that should be processed/used by the protocol

        Parameters
        ----------
        variable_filepath (str): path to the file that contains the variable of interest
        variable_name (str): name of the variable of interest in self.morphoframe
        morphoframe (bool): choose betwenn self.morphoframe and self.metadata
        
        Returns
        -------
        _morphoframe (dict): the variable used by the protocol
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
        protocol_name (str): the name of the protocol
        save_folderpath (str): the folder path containing the saved file 
        save_filename (str): name of the file that will be saved
        save_data (bool): data will be saved or not

        Returns
        -------
        save_filepath (str): the path of the file containing the output of the protocol
        update save_folderpath and save_filename of the protocol in self.parameters
        """
        if save_data:
            if save_filename == 0:
                self.parameters[protocol_name]["save_filename"] = default_save_filename
            else:
                self.parameters[protocol_name]["save_filename"] = self.file_prefix + '.' + save_filename
            if save_folderpath == 0:
                self.parameters[protocol_name]["save_folderpath"] = os.getcwd()
            save_filepath = "%s/%s" % (self.parameters[protocol_name]["save_folderpath"], self.parameters[protocol_name]["save_filename"])
        else:
            save_filepath = None

        return save_filepath
    
    def _image_filtering(self, persistence_images, params, save_filepath):

        #if os.path.isfile(params["FilteredPixelIndex_filepath"]):
        if params["FilteredPixelIndex_filepath"]:
            print("Loading indices used for filtering persistence images...")
            _tokeep = morphomics.utils.load_obj(params["FilteredPixelIndex_filepath"].replace(".pkl", ""))
        else:
            print("Keeping pixels in persistence image with standard deviation of %.3f..."%float(params["pixel_std_cutoff"]))
            _tokeep = np.where(
                np.std(persistence_images, axis=0) >= params["pixel_std_cutoff"]
            )[0]
            
        filtered_image = np.array([np.array(pi[_tokeep]) for pi in persistence_images])
        self.metadata["pixes_tokeep"] = _tokeep

        if params["save_data"]:
            print("The filtration is saved in %s" %(save_filepath))
            morphomics.utils.save_obj(self.metadata, "%s-FilteredIndex" % (save_filepath))
            morphomics.utils.save_obj(self.morphoframe[params["morphoframe_name"]], "%s-FilteredMatrix" % (save_filepath))
            
        return filtered_image


    ## Public
    def Input(self):
        """
        Protocol: Load .swc files, transform them into TMD barcodes and store them as a morphoframe.
        
        Essential parameters:
            data_location_filepath (str): location of the filepath
            extension (str): .swc file extension, "_corrected.swc" refers to .swc files that were corrected with NeurolandMLConverter
            conditions (list, str): this must match the hierarchical structure of `data_location_filepath`
            separated_by (str): saving chunks of the morphoframe via this condition, this must be an element of `conditions`
            filtration_function (str): this is the TMD filtration function, can either be radial_distances, or path_distances
            morphoframe_name (str): this is how the morphoframe will be called
            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            save_filename (str or 0): this will be used as the file name
        
        Returns
        -------
        Add a dataframe with key morphoframe_name to morphoframe.
        Each row in the dataframe is data from one cell and each column is a fetaure of the cell.
        """

        params = self.parameters["Input"]

        data_location_filepath = params["data_location_filepath"]
        extension = params["extension"]
        conditions = params["conditions"]
        separated_by = params["separated_by"]
        filtration_function = params["filtration_function"]
        morphoframe_name = params["morphoframe_name"]
        
        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]
        
        # define output filename
        default_save_filename = "%s.TMD-%s"%(self.file_prefix, filtration_function)
        save_filepath = self._set_filename(protocol_name = "Input", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)

        print("Loading the data from %s"%(data_location_filepath))
        print("Saving dataset in %s"%(save_filepath))

        # load the data
        self.morphoframe[morphoframe_name] = morphomics.io.load_data(
            folder_location = data_location_filepath,
            extension = extension,
            conditions = conditions,
            filtration_function = filtration_function,
            separated_by = separated_by,
            save_filename = save_filepath,
        )
        

        print("Input done!")
        print("")



    def Load_data(self):
        """
        Protocol: Load morphoframe from .pkl files.
        
        Essential parameters:
            folderpath_to_data (str): location to the pickle file outputs to Protocols.Input 
            filepath_to_data (0 or str): full path to file to be loaded
            conditions_to_include (list of str): the different conditions that you want to load
            morphoframe_name (str): this is how the morphoframe will be called
        
        Returns
        -------
        Add a dataframe to morphoframe.
        """
        params = self.parameters["Load_data"]
        
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
        
        

    def Clean_frame(self):
        """
        Protocol: Clean out the morphoframe, to filter out artifacts and unwanted conditions.
        
        Essential parameters:
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be filtered out
            barcode_size_cutoff (int): remove morphologies if the number of bars is less than the cutoff
            barlength_cutoff (list, (str, float)): retain bars whose length satisfy a certain cutoff
                                    must be an array with two elements, [">" "<", ">=", "<=", "==", bar length cutoff]
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
        params = self.parameters["Clean_frame"]
        
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
        default_save_filename = "%s.Cleaned"%(self.file_prefix)
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

        
        
    def Bootstrap(self):
        """
        Protocol: Takes the morphoframe and bootstraps the variable specified in  `feature_to_bootstrap` and returns a new morphoframe with the bootstrapped samples
        
        Essential parameters:
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be filtered out
            feature_to_bootstrap (list, (str, str)): a column in morphoframe and the type (either bars, scalar or array) to bootstrap
            bootstrap_conditions (list, str): if you want to bootstrap over all the conditions in "morphoframe_name", then leave this as is and leave the "bootstrap_conditions" empty
            rand_seed (int): seed of the random number generator
            N_bags (int): number of morphologies to take averages of
            n_samples (int): number of bootstrap samples to create
            ratio (float): a number between 0 and 1, if this is opted, N_pop will be calculated as ratio*(total number of morphologies in a given condition combination)
            bootstrapframe_name (str): where the bootstrapped morphoframes will be stored
            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            default_save_filename (str or 0): this will be used as the file prefix

        Returns
        -------
        Add a dataframe containing bootstrapped data to morphoframe. The samples are sub groups of microglia.
        """
        params = self.parameters["Bootstrap"]

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
        
        print("Bootstrapping with the following parameters: ")
        print("bootstrap resolution: %s"%("-".join(bootstrap_conditions)))
        print("bag size: %d"%(int(n_samples)))
        print("number of bootstrap bags: %d"%(int(N_bags)))
        bootstrapped_frame = (
            morphomics.bootstrapping.get_bootstrap_frame(
                _morphoframe,
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
        default_save_filename = "%s.Bootstrapped"%(self.file_prefix)
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



    def Vectorizations(self):
        """
        Protocol: Takes a morphoframe called morphoframe_name and vectorize the barcodes within the morphoframe
        
        Essential parameters:
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be filtered out
            vect_method_parameters (dict): keys are the vectorization methods applied on the TMD and attributes are the parameters of each vetorization method
            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            save_filename (str or 0): this will be used as the file name

        Returns
        -------
        Add a list of vectors to a morphoframe. A row per sample, the colums are the dimensions of the vector (result of the TMD vectorization).
        """

        params = self.parameters["Vectorizations"]

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]
        vect_method_parameters = params["vect_method_parameters"]
        
        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        # define morphoframe containing barcodes to compute vectorizations
        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                           variable_name = morphoframe_name)
        
        assert (
            "barcodes" in _morphoframe.keys()
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
        vectorizer = Vectorizer(tmd = _morphoframe["barcodes"], 
                                vect_parameters = vect_method_parameters)
        
        # compute vectors
        output_vectors = []
        for vect_method in vect_methods:
            perform_vect_method = getattr(vectorizer, vect_method)
            output_vector = perform_vect_method()
            
            output_vectors.append(output_vector)
        output_vectors = np.concatenate(output_vectors, axis=1)

        # define output filename
        default_save_filename = "%s.Vectorizations-%s"%(self.file_prefix, vect_methods_codename)
        save_filepath = self._set_filename(protocol_name = "Vectorizations", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)
        
        self.morphoframe[morphoframe_name] = _morphoframe
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
            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            save_filename (str or 0): this will be used as the file name

        Returns
        -------
        Add a list of reduced vectors to a morphoframe. A row per sample (example: microglia), the colums are the dimensions of the reduced vectors (result of the dimensionality reduction).
        """
        params = self.parameters["Dim_reductions"]

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]
        dimred_method_parameters = params["dimred_method_parameters"]
        vectors_to_reduce = params["vectors_to_reduce"]
        filter_pixels = params["filter_pixels"]
        normalize = params['normalize']

        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        # define morphoframe containing vectors to compute dim reduction
        _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                           variable_name = morphoframe_name)
        
        X = np.vstack(_morphoframe[vectors_to_reduce])
        
        # if persistence image, pixels can be filtered 
        if filter_pixels:
            filtered_image = self._image_filtering(persistence_images = X,
                                                  params = params, 
                                                  save_filename = save_filename)
            X = filtered_image

        # normalize data 
        if normalize:
            print("Normalize the vectors")
            normalizer = StandardScaler()
            X = normalizer.fit_transform(X)

        dimred_methods = dimred_method_parameters.keys()
        dimred_method_names = '_'.join(list(dimred_methods))

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

        # define output filename
        default_save_filename = "%s.DimReductions-%s"%(self.file_prefix, dimred_method_names)
        save_filepath = self._set_filename(protocol_name = "Dim_reductions", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, 
                                              save_data = save_data)

        self.morphoframe[morphoframe_name] = _morphoframe
        self.morphoframe[morphoframe_name][dimred_method_names] = list(reduced_vectors)

        # save the reduced vectors
        if save_filename is not None:
            print("The reduced vectors and fitted dimreducers are saved in %s" %(save_filepath))

            save_obj(obj = self.morphoframe[morphoframe_name], filepath = save_filepath + '_reduced_data')
            save_obj(obj = self.metadata, filepath = save_filepath + '_fitted_dimreducer')

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
        
        Essential parameters:
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
        params = self.parameters["Save_reduced"]

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
        default_save_filename = "%s.ReductionInfo"%(self.file_prefix)
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
            save_data (bool): trigger to save output of protocol
            save_folderpath (str): location where to save the data
            save_filename (str or 0): this will be used as the file name

        Returns
        -------
        Add a list of reduced vectors to a morphoframe. A row per sample (example: microglia), the colums are the dimensions of the reduced vectors (result of the dimensionality reduction).
        """
        params = self.parameters["Mapping"]

        fitted_dimreducer_filepath = params["fitted_dimreducer_filepath"]
        dimred_method = params["dimred_method"]

        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]
        vectors_to_reduce_name = params["vectors_to_reduce_name"]

        filter_pixels = params["filter_pixels"]
        FilteredPixelIndex_filepath = params["FilteredPixelIndex_filepath"]
        
        normalize = params['normalize']
        
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
        default_save_filename = "%s.Mapping"%(self.file_prefix)
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
            normalizer = StandardScaler()
            vectors_to_reduce = normalizer.fit_transform(vectors_to_reduce)

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

        """
        params = self.parameters["Plotting"]
            
        morphoframe_filepath = params["morphoframe_filepath"]
        morphoframe_name = params["morphoframe_name"]
        conditions = params['conditions']
        reduced_vectors_name = params["reduced_vectors_name"]
        axis_labels = params['axis_labels']
        title = params['title']
        colors = params['colors']
        size= params['size']
        amount = params['amount']

        save_data = params["save_data"]
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"] 

        # define morphoframe that contains the data points to plot
        print("Loading fitted dim reduction function...")   

        if type(morphoframe_filepath) is str:
            if "csv" in morphoframe_filepath:
                _morphoframe = pd.read_csv(morphoframe_filepath,index_col = 0)
        else:
            _morphoframe = self._get_variable(variable_filepath = morphoframe_filepath,
                                                variable_name = morphoframe_name)
            # one column per coordinate
            _morphoframe = _morphoframe.copy()
            reduced_vectors = _morphoframe[reduced_vectors_name].copy()
            reduced_vectors = np.vstack(reduced_vectors)
            for dims in range(reduced_vectors.shape[1]):
                _morphoframe[axis_labels[dims]]  = reduced_vectors[:, dims]


        fig3d = plotting.plot_3d_scatter(morphoframe = _morphoframe,
                                 axis_labels = axis_labels,
                                 conditions = conditions,
                                 colors = colors,
                                 amount= amount,
                                 size = size,
                                 title = title)
        
        fig2d = plotting.plot_2d_scatter(morphoframe = _morphoframe,
                                 axis_labels = axis_labels,
                                 conditions = conditions,
                                 colors = colors,
                                 amount= amount,
                                 size = size,
                                 title = title)
        
        # define output filename
        default_save_filename = "%s.Plotting"%(self.file_prefix)
        save_filepath = self._set_filename(protocol_name = "Plotting", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename)

        if save_data:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
            # Save the plot as an HTML file
            fig3d.write_html(save_filepath + '3d.html')
            fig3d.write_image(save_filepath + '3d.pdf', format = 'pdf')
            fig2d.write_html(save_filepath + '2d.html')
            fig2d.write_image(save_filepath + '2d.pdf', format = 'pdf')
            print(f"Plot saved as {save_filepath}")
        print("Plotting done!")
        print("")



    def Save_parameters(self):
        # Save a dictionary containing the name of the processed data and the parameters of the main steps for reproducibility
        params = self.parameters['Save_parameters']
        save_folderpath = params["save_folderpath"]
        save_filename = params["save_filename"]

        default_save_filename = "%s.Experiment_Parameters"%(self.file_prefix)
        save_filepath = self._set_filename(protocol_name = "Save_parameters", 
                                              save_folderpath = save_folderpath, 
                                              save_filename = save_filename,
                                              default_save_filename = default_save_filename, )
        morphomics.utils.save_obj(obj = params,
                                    filepath = save_filepath) 
        print("The experiment parameters are saved in %s" %(save_filepath))
        print("")



    def Clear_morphoframe(self):
        """
        Protocol: Clears morphoframe
        """
        print("Clearing morphoframe...")
        self.morphoframe = {}