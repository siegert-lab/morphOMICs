import morphomics
import morphomics.io

from morphomics.Analysis.vectorizer import Vectorizer
from morphomics.Analysis.embedder import Embedder

from morphomics.utils import save_obj, vectorization_codenames
from sklearn.preprocessing import Normalizer

import numpy as np
import pandas as pd
import os
import umap
from sklearn.decomposition import PCA
import ipyvolume as ipv  # https://ipyvolume.readthedocs.io/en/latest/install.html
from matplotlib import cm, colors
import matplotlib.pyplot as plt

class Protocols(object):
    
    def __init__(self, parameters, Parameters_ID):
        self.parameters = parameters
        self.file_prefix = f"Morphomics.PID_{Parameters_ID}"
        self.morphoframe = {}
        self.metadata = {}
        
        print("Unless you have specified the file prefix in the succeeding executables, \nthis will be the file prefix: %s"%(self.file_prefix))

    ## Private
    def _find_variable(self, variable_filepath, variable_name):
        """
        finds the variable that should be processed/used by the protocol

        Parameters
        ----------
        variable_filepath (str): path to the file that contains variable of interest
        variable_name (str): name of the variable of interest in morphoframe

        Returns
        -------
        the variable used by the protocol
        """ 
        if variable_filepath:
            print("Loading %s file..." %(variable_filepath))
            _morphoframe = morphomics.utils.load_obj(variable_filepath.replace(".pkl", ""))
        else:
            _morphoframe = self.morphoframe[variable_name]  
        
        return _morphoframe

    def _define_filename(self, params, save_folder_path, file_prefix, save_data = True):
        """
        defines the name of the file containing the output of the protocol

        Paramters
        ---------
        params (dict): the parameters of the protocol
        save_folder_path (str): the folder path to where file is saved 
        file_prefix (str): a big part of the name of the file to save
        save_data (bool): data will be saved or not

        Returns
        -------
        the name of the file containing the output of the protocol
        """
        if save_data:
            if params["file_prefix"] == 0:
                params["file_prefix"] = file_prefix
            if save_folder_path == 0:
                params["save_folder"] = os.getcwd()
            save_filename = "%s/%s" % (params["save_folder"], params["file_prefix"])
        else:
            save_filename = None

        return save_filename
    
    def _image_filtering(self, persistence_images, params, save_filename):

        if params["filteredpixelindex_filepath"]:
            print("Loading indices used for filtering persistence images...")
            _tokeep = morphomics.utils.load_obj(params["filteredpixelindex_filepath"].replace(".pkl", ""))
        else:
            print("Keeping pixels in persistence image with standard deviation of %.3f..."%float(params["pixel_std_cutoff"]))
            _tokeep = np.where(
                np.std(persistence_images, axis=0) >= params["pixel_std_cutoff"]
            )[0]
            
        filtered_image = np.array([np.array(pi[_tokeep]) for pi in persistence_images])

        if params["save_data"]:
            morphomics.utils.save_obj(_tokeep, "%s-FilteredIndex" % (save_filename))
            morphomics.utils.save_obj(self.metadata["PI_matrix"], "%s-FilteredMatrix" % (save_filename))
            
        return filtered_image


    ## Public
    def Input(self):
        """
        Protocol: Load .swc files, transform them into TMD barcodes and store them as a morphoframe
        
        Essential parameters:
            data_location_filepath (str): location of the filepath
            extension (str): .swc file extension, "_corrected.swc" refers to .swc files that were corrected with NeurolandMLConverter
            filtration_function (str): this is the TMD filtration function, can either be radial_distances, or path_distances
            conditions (list, str): this must match the hierarchical structure of `data_location_filepath`
            separated_by (str): saving chunks of the morphoframe via this condition, this must be an element of `conditions`
            morphoframe_name (str): this is how the morphoframe will be called
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix

        Returns
        -------
        add a dataframe with key morphoframe_name to morphoframe
        each row in the dataframe is tmd data from one cell
        """
        params = self.parameters["Input"]

        
        # define output filename
        file_prefix = "%s.TMD-%s"%(self.file_prefix, params["filtration_function"])
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])

        print("Loading the data from %s"%(params["data_location_filepath"]))
        print("Saving dataset in %s"%(save_filename))

        # load the data
        self.morphoframe[params["morphoframe_name"]] = morphomics.io.load_data(
            folder_location=params["data_location_filepath"],
            extension=params["extension"],
            conditions=params["conditions"],
            filtration_function=params["filtration_function"],
            separated_by=params["separated_by"],
            save_filename=save_filename,
        )

        print("Input done!")



    def Load_data(self):
        """
        Protocol: Load morphoframe output files from Protocols.Input
        
        Essential parameters:
            folderpath_to_data (str): location to the pickle file outputs to Protocols.Input 
            filename_prefix (str): ommon prefix of the pickle files to be loaded, must be inside `folderpath_to_data`
            conditions_to_include (arr): the different conditions that you want to load
            morphoframe_name (str): this is how the morphoframe will be called
        """
        params = self.parameters["Load_data"]
        
        # _morphoframe = {}
        # for _c in params["conditions_to_include"]:
        #     print("...loading %s" % _c)
        #     save_filename = "%s/%s%s" % (params["folderpath_to_data"], params["filename_prefix"], _c)
        #     _morphoframe[_c] = morphomics.utils.load_obj(save_filename.replace(".pkl", ""))

        # self.morphoframe[params["morphoframe_name"]] = pd.concat([_morphoframe[_c] for _c in params["conditions_to_include"]], ignore_index=True)
        self.morphoframe[params["morphoframe_name"]] = morphomics.utils.load_obj(params["folderpath_to_data"])
        
        

    def Clean_frame(self):
        """
        Protocol: Clean out the morphoframe, to filter out artifacts and unwanted conditions
        
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
        """
        params = self.parameters["Clean_frame"]
        
        # define morphoframe to clean
        _morphoframe = self._find_variable(variable_filepath = params["morphoframe_filepath"],
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
        file_prefix = "%s.Cleaned"%(self.file_prefix)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])
        
        # change file name prefix
        # save the file 
        # TODO armonizer la sauvegarde
        if params["save_data"]:
            if params["file_prefix"] == 0:
                self.file_prefix = file_prefix

            morphomics.utils.save_obj(_morphoframe, save_filename)
            
        self.morphoframe[params["morphoframe_name"]] = _morphoframe

        
        
    def Bootstrap(self):
        """
        Protocol: Takes the morphoframe and bootstraps the variable specified in  `feature_to_bootstrap` and returns a new morphoframe with the bootstrapped samples
        
        Essential parameters:
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be filtered out
            feature_to_bootstrap (list, (str, str)): a column in morphoframe and the type (either bars, scalar or array) to bootstrap
            condition_column (str): column name in morphoframe where the `bootstrap_conditions` are located
            bootstrap_conditions (list, str): if you want to bootstrap over all the conditions in "morphoframe_name", then leave this as is and leave the "bootstrap_conditions" empty
            bootstrap_resolution (list ,str): which conditions combinations which bootstrapping will consider as a unique condition restrictions
            rand_seed (int): seed of the random number generator
            ratio (float): a number between 0 and 1, if this is opted, N_pop will be calculated as ratio*(total number of morphologies in a given condition combination)
            N_pop (int): number of morphologies to take averages of
            N_samples (int): number of bootstrap samples to create
            bootstrapframe_name (str): where the bootstrapped morphoframes will be stored
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Bootstrap"]
                
        # initialize morphoframe to bootstrap
        _morphoframe = self._find_variable(variable_filepath = params["morphoframe_filepath"],
                                            variable_name = params["morphoframe_name"])   

        # define output filename
        file_prefix = "%s.Bootstrapped"%(self.file_prefix)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])
            
        if params["ratio"] == 0:
            ratio = None
        
        print("Bootstrapping with the following parameters: ")
        print("bootstrap resolution: %s"%("-".join(params["bootstrap_resolution"])))
        print("bootstrap size: %d"%(int(params["N_pop"])))
        print("number of bootstraps: %d"%(int(params["N_samples"])))
        bootstrapped_frame = (
            morphomics.bootstrapping.get_subsampled_population_from_infoframe(
                _morphoframe,
                feature_to_bootstrap=params["feature_to_bootstrap"],
                condition_column=params["condition_column"],
                bootstrap_conditions=params["bootstrap_conditions"],
                bootstrap_resolution=params["bootstrap_resolution"],
                N_pop=params["N_pop"],
                N_samples=params["N_samples"],
                rand_seed=params["rand_seed"],
                ratio=ratio,
                save_filename=save_filename,
            )
        )
        print("Bootstrap done!")
        
        self.metadata[params["morphoinfo_name"]] = (
            bootstrapped_frame[params["bootstrap_resolution"]]
            .reset_index(drop=True)
            .astype("category")
        )
        
        self.morphoframe[params["bootstrapframe_name"]] = bootstrapped_frame
        
        if save_filename is not None:
            save_obj(self.morphoframe[params["bootstrapframe_name"]], save_filename)
            save_obj(self.metadata[params["morphoinfo_name"]], "%s-MorphoInfo"%save_filename)
            


    def Persistence_Images(self):
        """
        Protocol: Takes a morphoframe in `morphoframe_name` and calculates the persistence images with the barcodes within the morphoframe
        
        Essential parameters:
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be filtered out
            bw_method (float): spread of the Gaussian kernel
            norm_method (str): how to normalize the persistence image, can be "sum" or "max"
            xlims (list, (float, float))): constraints to the x-axis limits
            ylims (list, (float, float))): constraints to the y-axis limits
            barcode_weight (float): key in the metadata which will be used to weight each bar
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Persistence_Images"]
        
        # define morphoframe to compute persistence image
        _morphoframe = self._find_variable(variable_filepath = params["morphoframe_filepath"],
                                           variable_name = params["morphoframe_name"])
        assert (
            "barcodes" in _morphoframe.keys()
        ), "Missing `barcodes` column in info_frame..."

        # define output filename
        file_prefix = "%s.PersistenceImages"%(self.file_prefix)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])
        
        # this is feature that will be developed in the future
        if params["barcode_weight"] == "None":
            params["barcode_weight"] = None
            
        print("Calculating persistence images with the following parameters:")
        print("Gaussian kernel size: %.3f"%(float(params["bw_method"])))
        print("image normalization method: %s"%(params["norm_method"]))
        
        vect_method = "persistence_image"
        pi_parameters = {vect_method: { "rescale_lims" : params["rescale_lims"],
                                        "xlims" : params["xlims"],
                                        "ylims" : params["ylims"],
                                        "bw_method" : params["bw_method"],
                                        "barcode_weight" : params["barcode_weight"],
                                        "norm_method" : params["norm_method"],
                                        "resolution" : params["resolution"]
                                        } 
                        }
        
        pi_vectorizer = Vectorizer(tmd = _morphoframe["barcodes"], 
                                              vect_parameters = pi_parameters)
        
        pis = pi_vectorizer.persistence_image()
 
        # save the persistence images
        if save_filename is not None:
            save_obj(obj = pis, filepath = save_filename)

        self.self.morphoframe[params["morphoframe_name"]]["PI_matrix"] = pis
        print("Persistence image done!")
        


    def Vectorizations(self):
        """
        Protocol: Takes a morphoframe in `morphoframe_name` and vectorize the TMD within the morphoframe
        
        Essential parameters:
            morphoframe_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            morphoframe_name (str): morphoframe key which will be filtered out
            vect_method_parameters (dict): keys are the vectorization methods applied on the TMD and attributes are the parameters of each vetorization method
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix

        Returns
        -------
        A numpy array of vectors. The rows are the samples (microglia), the colums are the features of the vector (result of the TMD vectorrization).
        """

        params = self.parameters["Vectorizations"]

        vect_methods = params["vect_method_parameters"].keys()
        vect_parameters = params["vect_method_parameters"]
        vect_methods_names = [vectorization_codenames[vect_method] for vect_method in vect_methods]
        vect_method_names = '_'.join(vect_methods_names)


        # define morphoframe to compute vectorizations
        _morphoframe = self._find_variable(variable_filepath = params["morphoframe_filepath"],
                                           variable_name = params["morphoframe_name"])
        assert (
            "barcodes" in _morphoframe.keys()
        ), "Missing `barcodes` column in info_frame..."

        # define output filename
        file_prefix = "%s.Vectorizations-%s"%(self.file_prefix, vect_method_names)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])

        print("Computes %s and concatenates the vectors" %(vect_method_names))
        
        vectorizer = Vectorizer(tmd = _morphoframe["barcodes"], 
                                            vect_parameters = vect_parameters)
        
        output_vectors = []
        for vect_method in vect_methods:
            perform_vect_method = getattr(vectorizer, vect_method)
            output_vector = perform_vect_method()
            
            output_vectors.append(output_vector)

        output_vectors = np.concatenate(output_vectors, axis=1)

        # save the output vectors
        if save_filename is not None:
            save_obj(obj = output_vectors, filepath = save_filename)

        self.morphoframe[params["morphoframe_name"]][vect_method_names] = list(output_vectors)
        print("Vectorization done!")



    def Embedding(self):
        """
        Protocol: Takes the vectors you want in morphoframe and embeds them following the chosen embedding techniques
        
        Parameters:
        ----------        
        Essential parameters:
            PersistenceImages_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            filter_pixels (str): morphoframe key which will be filtered out
            filteredpixelindex_filepath (float): spread of the Gaussian kernel
            pixel_std_cutoff (str): how to normalize the persistence image, can be "sum" or "max"
            run_PCA (1 or 0)): if 0, the persistence image array will not be initially reduced
            n_PCs (list, (float, float))): number of PCs
            n_neighbors (int): how many neighbors to consider when constructing the connectivity matrix
            min_dist (float): minimum distance between points to consider as neighbors
            spread (float): spread of points in the UMAP manifold
            metric (str): metric to use to calculate distances between points
            random_state (int): seed of the random number generator
            densmap (1 or 0): trigger density-preserving mapping, i.e., location of points in the UMAP manifold reflects sparsity of points
            n_components (int): dimension of the UMAP manifold
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_

        """
        params = self.parameters["Embedding"]

        embed_methods = params["embed_method_parameters"].keys()
        embed_parameters = params["embed_method_parameters"]
        embed_method_names = '_'.join(list(embed_methods))

        vectors_to_embed = params["vectors_to_embed"]

        # define morphoframe to compute embedding
        _morphoframe = self._find_variable(variable_filepath = params["morphoframe_filepath"],
                                           variable_name = params["morphoframe_name"])
        X = np.vstack(_morphoframe[vectors_to_embed])

        # define output filename
        file_prefix = "%s.Embeddings-%s"%(self.file_prefix, embed_method_names)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])
        
        if params["filter_pixels"]:
            filtered_image = self._image_filtering(persistence_images = X,
                                                  params = params, 
                                                  save_filename = save_filename)
            X = filtered_image

        normalize = params['normalize']
        if normalize:
            print("Normalize the vectors")
            normalizer = Normalizer()
            X = normalizer.fit_transform(X)

        print("Embeds the vectors with the following techniques %s " %(embed_method_names))
        embedder = Embedder(tmd_vectors = X,
                            embed_parameters = embed_parameters)
        
        fit_embedders = []
        for embed_method in embed_methods:
            perform_embed_method = getattr(embedder, embed_method)
            fit_embedder, embedded_vectors = perform_embed_method()

            self.metadata[embed_method] = fit_embedder
            fit_embedders.append(fit_embedder)
   
            embedder.tmd_vectors = embedded_vectors

        # save the embedded vectors
        if save_filename is not None:
            save_obj(obj = embedded_vectors, filepath = save_filename + '_embedded_data')
            save_obj(obj = fit_embedders, filepath = save_filename + '_fitted_embedder')

        self.morphoframe[params["morphoframe_name"]][embed_method_names] = list(embedded_vectors)
        print("Embedding done!")



    def Palantir(self):
        """
        Protocol: Takes the UMAP manifold, calculates diffusion maps using Palantir and outputs a force-directed layout of the maps
        
        Essential parameters:
            X_umap_filepath (str or 0): if not 0, must contain the filepath to the morphoframe which will then be saved into morphoframe_name
            n_diffusion_components (int): number of diffusion maps to generate
            knn_diffusion (int): number of nearest neighbors that will be used to generate diffusion maps
            fdl_random_seed (float): seed of the random number generator for the force-directed layout
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Palantir"]
        
        # define output filename
        file_prefix = "%s.Palantir"%(self.file_prefix)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])
            
        if params["X_umap_filepath"]:
            print("Loading UMAP coordinates...")
            self.metadata["X_umap"] = morphomics.utils.load_obj(params["X_umap_filepath"].replace(".pkl", ""))
        
        print("Calculating diffusion maps with Palantir...")
        self.metadata["palantir_distances"] = morphomics.reduction.palantir_diffusion_maps(
            self.metadata["X_umap"], 
            n_components=params["n_diffusion_components"], 
            knn=params["knn_diffusion"],
        )

        print("Calculating 2D coordinates with force-directed layout")
        self.metadata["X_fdl"] = morphomics.reduction.force_directed_layout(
            self.metadata["palantir_distances"]["kernel"], 
            random_seed=params["fdl_random_seed"]
        )

        print("Palantir done!")
        
        if params["save_data"]:
            morphomics.utils.save_obj(self.metadata["palantir_distances"], "%s-PalantirDistances" % (save_filename) )
            morphomics.utils.save_obj(self.metadata["X_fdl"], "%s-PalantirFDCoords" % (save_filename) )
        
        
        
    def old_Prepare_ReductionInfo(self):
        """
        Protocol: Takes the UMAP manifold coordinates and conditions to create a .csv file which can be uploaded to the morphOMICs dashboard
        
        Essential parameters:
            declare_filepaths (true or false): prompt whether UMAP and morphoinfo files will be declared
            embedded_data_filepath (str): filepath to the UMAP manifold coordinates
            BootstrapInfo_filepath (str): filepath to the morphoinfo file corresponding to the UMAP manifold coordinates
            coordinate_key (str): metadata key where UMAP coordinates are located, or will be stored
            morphoinfo_key (str): metadata key where morphoinfo is located, or will be stored
            coordinate_axisnames (str): name of the coordinate (e.g., UMAP)
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Prepare_ReductionInfo"]
    
        # define output filename
        file_prefix = "%s.ReductionInfo"%(self.file_prefix)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])
        save_filename = "%s.csv" % (save_filename)

        print("Preparing .csv file that can be loaded into the morphOMICs dashboard...")
        
        if params["declare_filepaths"]:
            self.metadata[params["coordinate_key"]] = morphomics.utils.load_obj( params["embedded_data_filepath"] )
            self.metadata[params["morphoinfo_key"]] = morphomics.utils.load_obj( params["BootstrapInfo_filepath"] )
                
            assert params["coordinate_key"] in self.metadata.keys(), "Run UMAP first!"
            assert params["morphoinfo_key"] in self.metadata.keys(), "Bootstrap_info is not found"
        
        _reduction_info = self.metadata[params["morphoinfo_key"]].copy()
        _reduction_info = self.metadata[params["coordinate_key"]].shape[1]
        print(_reduction_info)
        for dims in range(self.metadata[params["coordinate_key"]].shape[1]):
            _reduction_info["%s_%d"%(params["coordinate_axisnames"], dims+1)] = self.metadata[params["coordinate_key"]][:, dims]
        _reduction_info.to_csv( save_filename )

        print("Reduction done!")
            
    def Prepare_ReductionInfo(self):
        """
        Protocol: Takes the UMAP manifold coordinates and conditions to create a .csv file which can be uploaded to the morphOMICs dashboard
        
        Essential parameters:
            declare_filepaths (true or false): prompt whether UMAP and morphoinfo files will be declared
            embedded_data_filepath (str): filepath to the UMAP manifold coordinates
            BootstrapInfo_filepath (str): filepath to the morphoinfo file corresponding to the UMAP manifold coordinates
            coordinate_key (str): metadata key where UMAP coordinates are located, or will be stored
            morphoinfo_key (str): metadata key where morphoinfo is located, or will be stored
            coordinate_axisnames (str): name of the coordinate (e.g., UMAP)
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Prepare_ReductionInfo"]
    
        # define output filename
        file_prefix = "%s.ReductionInfo"%(self.file_prefix)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])
        save_filename = "%s.csv" % (save_filename)

        print("Preparing .csv file that can be loaded into the morphOMICs dashboard...")

        _morphoframe = self._find_variable(variable_filepath = params["morphoframe_filepath"], 
                                           variable_name = params["morphoframe_name"])
        
        reduction_morphoframe = _morphoframe.copy()
        print(_morphoframe[params["coordinate_key"]].shape())
        for dims in range(_morphoframe[params["coordinate_key"]].shape[1]):
            reduction_morphoframe["%s_%d"%(params["coordinate_axisnames"], dims+1)] = reduction_morphoframe[params["coordinate_key"]][:, dims]

        reduction_morphoframe.to_csv( save_filename )

        print("Reduction done!")
        
    def Mapping(self):
        """
        Protocol: Takes a pre-calculated UMAP function, maps persistence images into the UMAP manifold and outputs the manifold coordinates
        
        Essential parameters:
            F_umap_filepath (str): location to the UMAP function that will be used 
            PersistenceImages_filepath (str or 0): if not 0, you MUST specify the location to the persistence images
            filter_pixels (1 or 0): prompt to filter pixels in the persistence images
            FilteredPixelIndex_filepath (str): location of the filtered pixel indices before doing the UMAP of the generated phenotypic spectrum
            run_PCA (1 or 0): prompt to take a pre-calculated PC reduction and map the persistence images into the PC space
            F_PCA_filepath (str): location of the PCA function
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Mapping"]

        # define output filename
        file_prefix = "%s.Mapping"%(self.file_prefix)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])            
            
        if os.path.isfile(params["F_umap_filepath"]):
            print("Loading UMAP function...")
            F_umap = morphomics.utils.load_obj(params["F_umap_filepath"].replace(".pkl", ""))
        else:
            print("!!! IMPORTANT !!!")
            print("Please provide the filepath to the UMAP function that generated the morphological spectrum.")
            exit()
            

        if params["PersistenceImages_filepath"]:
            print("Loading persistence image matrix file...")
            self.metadata["PI_matrix"] = morphomics.utils.load_obj(params["PersistenceImages_filepath"].replace(".pkl", ""))
            
        if params["filter_pixels"]:
            if os.path.isfile(params["FilteredPixelIndex_filepath"]):
                print("Loading indices used for filtering persistence images...")
                _tokeep = morphomics.utils.load_obj(params["FilteredPixelIndex_filepath"].replace(".pkl", ""))
            else:
                print("!!! IMPORTANT !!!")
                print("It is important that the indices filtered in the persistence image is consistent with that in the generated morphological spectrum.")
                exit()
                
            print("Filtering persistence images...")
            self.metadata["PI_matrix"] = np.array([np.array(self.metadata["PI_matrix"][_i][_tokeep]) for _i in np.arange(len(self.metadata["PI_matrix"]))])

            if params["save_data"]:
                morphomics.utils.save_obj(self.metadata["PI_matrix"], "%s-FilteredMatrix" % (save_filename))
                
        if params["run_PCA"]:
            if os.path.isfile(params["F_PCA_filepath"]):
                print("Loading PCA function...")
                F_PCA = morphomics.utils.load_obj(params["F_PCA_filepath"].replace(".pkl", ""))
            else:
                print("!!! IMPORTANT !!!")
                print("It is important that the PCA function is consistent with that in the generated morphological spectrum.")
                exit()    
    
            print("Transforming matrix into PC-space...")
            self.metadata["PI_matrix"] = F_PCA.transform(self.metadata["PI_matrix"])
            
            if params["save_data"]:
                morphomics.utils.save_obj(self.metadata["PI_matrix"], "%s-PCAcoords" % (save_filename))
        
        print("Mapping persistence images into the UMAP space...")
        self.metadata["X_umap"] = F_umap.transform(self.metadata["PI_matrix"])
        
        print("Mapping done!")
        
        if params["save_data"]:
            morphomics.utils.save_obj(self.metadata["X_umap"], "%s-UMAPcoords%dD" % (save_filename, self.metadata["X_umap"].shape[1]) )
        
       
        
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
        save_filename = self._define_filename(params = params, 
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
        
        if params["save_data"]:
            morphomics.utils.save_obj(self.metadata[params["Sholl_colname"]], "%s" % (save_filename) )
            
            
            
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
        save_filename = self._define_filename(params = params, 
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
        
        if params["save_data"]:
            morphomics.utils.save_obj(self.metadata[params["Morphometric_colname"]], "%s" % (save_filename) )
            
            
            
    def Plotting(self):
        """
        Protocol: Generates a 3D interactive plot from the Protocols results, or from the  ReductionInfo files, or from coordinate and morphoinfo files
        
        Essential parameters:
            ReductionInfo_filepath (list, str)
            coordinate_key (str): metadata key where UMAP coordinates are located, or will be stored
            morphoinfo_key (str): metadata key where morphoinfo is located, or will be stored
            coordinate_axisnames (str): name of the coordinate (e.g., UMAP)
            Coordinate_filepath (list, str or 0): if these are not 0, must point to the location of the manifold coordinates
            MorphoInfo_filepath (list, str or 0): if these are not 0, must point to the location of the morpho_infoframe that corresponds to each element of `Coordinate_filepath`
            colormap_filepath (list, str): location to the color mapping that will be used
            label_prefixes (list ,str): prefixes to use for legend labels, must be same size as `colormap_filepath`
            Substitutions (list, (str, str, str)): ff you need to substitute the name of a condition in morpho_infoframe, use this
            show_plot (bool): trigger to show the interactive plot
            save_data (bool): trigger to save output of protocol
            save_folder (str): location where to save the data
            file_prefix (str or 0): this will be used as the file prefix
        """
        params = self.parameters["Plotting"]
        
        # define output filename
        file_prefix = "%s.Plot"%(self.file_prefix)
        save_filename = self._define_filename(params = params, 
                                              save_folder_path = params["save_folder"], 
                                              file_prefix = file_prefix, 
                                              save_data = params["save_data"])   
            
        assert len(params["colormap_filepath"]) > 0, "There must be a colormap_filepath!"
        assert np.sum([os.path.isfile(color_path) for color_path in params["colormap_filepath"]])==len(params["colormap_filepath"]), "Make sure that all files in colormap_filepath exists!"
        
        print("Creating the interactive plot...")
        ipv.figure(width=1920, height=1080)
        ipv.style.box_off()

        for _idx in range(len(params["label_prefixes"])):
            # load colormap
            colormap = pd.read_csv(params["colormap_filepath"][_idx], comment="#")
            colormap.Color = colormap.Color.str.split(";")
            colormap.GradientLimits = colormap.GradientLimits.str.split(";")

            # check if reductioninfo file is given
            if params["ReductionInfo_filepath"][_idx] == 0:
                coordinates = morphomics.utils.load_obj(
                    params["Coordinate_filepath"][_idx].replace(".pkl", "")
                )
                morpho_info = morphomics.utils.load_obj(
                    params["MorphoInfo_filepath"][_idx].replace(".pkl", "")
                )
                
            else:
                if os.path.exists(params["ReductionInfo_filepath"][_idx]):
                    morpho_info = pd.read_csv(params["ReductionInfo_filepath"][_idx], sep=",", header=0)
                    print(morpho_info.columns)
                    coordinate_columns = [
                        _cols
                        for _cols in morpho_info.columns
                        if params["coordinate_axisnames"] in _cols
                    ]
                    coordinates = np.array(morpho_info[coordinate_columns].values)
                else:
                    print("The following file was not found: %s"%params["ReductionInfo_filepath"][_idx])
                    print("Inferring the morpho_infoframe and manifold coordinates from Protocol class.")
                    
                    assert params["coordinate_key"] in self.metadata.keys(), "Cannot infer manifold coordinates. Run UMAP first!"
                    #assert params["morphoinfo_key"] in self.metadata.keys(), "Morpho_infoframe not found. Check this!"
                
                    morpho_info = self.metadata[params["morphoinfo_key"]]
                    coordinates = self.metadata[params["coordinate_key"]]

            conditions = [
                conds
                for conds in colormap.columns
                if conds not in ["Color_type", "Color", "GradientLimits"]
            ]

            try:
                for _cond, _before, _after in params["Substitutions"][_idx]:
                    morpho_info.loc[morpho_info[_cond] == _before, _cond] = _after
            except:
                print("No substitutions for coordinate set %d..."%(_idx+1))

            morphomics.plotting.scatterplot_3D_conditions(
                coordinates, morpho_info, conditions, colormap, params["label_prefixes"][_idx]
            )
            
            # the first file is considered as the main spectrum
            if _idx == 0:
                morphomics.plotting.scatterplot_3D_all(coordinates)

        ipv.xyzlabel("%s 1"%(params["coordinate_axisnames"]), 
                     "%s 2"%(params["coordinate_axisnames"]), 
                     "%s 3"%(params["coordinate_axisnames"]))

        if params["save_data"]:
            ipv.save(
                "%s-Spectrum.html" % (save_filename),
                title="Morphological spectrum",
                offline=False,
            )

        if params["show_plot"]:
            ipv.show()
        else:
            ipv.close()
            
            
            
    def Clear_morphoframe(self):
        """
        Protocol: Clears morphoframe
        """
        print("Clearing morphoframe...")
        self.morphoframe = {}