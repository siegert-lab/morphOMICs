save_folderpath = "br_experiments/results"

def parameters_br_pi_boot_umap_plot(save_folderpath = "br_experiments/results",
                                        pid = "all_br"):
        parameters = {}
        parameters["Parameters_ID"] = pid
        parameters["Protocols"] = [ "Input",
                                "Clean_frame",
                                "Bootstrap",
                                "Vectorizations",
                                "Dim_reductions",
                                "Save_reduced",
                                "Plotting",
                                ]


        # Input parameters
        parameters["Input"] = {"data_location_filepath" : "br_experiments/_Brain_Alessandro",
                                "extension" : "_corrected.swc",
                                "conditions" : ["Region",
                                                "Model",
                                                "Sex",
                                                "Animal"],  
                                "separated_by" : "Region",
                                "filtration_function" : "radial_distances",
                                "morphoframe_name" : "all_br",
                                "save_data" : True,
                                "save_folderpath" : save_folderpath + "/tmd",
                                "save_filename" : "tmd_morphoframe"
                                }


        # Clean_frame parameters
        parameters["Clean_frame"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : "all_br",
                                "barcode_size_cutoff" : 5,
                                "barlength_cutoff" : [],
                                "combine_conditions" : [],
                                "restrict_conditions" : [],
                                "save_data" : True,
                                "save_folderpath" : save_folderpath + "/tmd",
                                "save_filename" : "tmd_morphoframe_cleaned",
                                }


        # Bootstrap parameters
        parameters["Bootstrap"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : "all_br",
                                "feature_to_bootstrap" : ["barcodes", "bars"],         
                                "condition_column" : "condition",
                                "bootstrap_conditions" : [],
                                "bootstrap_resolution" : ["Region",
                                                        "Model",
                                                        "Sex"],
                                "rand_seed" : 34151,
                                "ratio" : 0,
                                "N_pop" : 15,
                                "N_samples" : 500,
                                "bootstrapframe_name" : "all_br_bootstrap",
                                "morphoinfo_name" : "bootstrap_info",
                                "save_data" : True,
                                "save_folderpath" : save_folderpath + "/bootstrap",
                                "save_filename" : "tmd_morphoframe_cleaned_bootstrap",
                                }


        # Vectorization parameters
        parameters["Vectorizations"] = {"morphoframe_filepath" : 0,
                                        "morphoframe_name" : "all_br_bootstrap",
                                        "save_data" : True,
                                        "save_folderpath" : save_folderpath + "/vectorized",
                                        "save_filename" : "tmd_morphoframe_cleaned_bootstrap_vect",
                                        "vect_method_parameters" :
                                                {'persistence_image' : {"rescale_lims" : False,
                                                                        "xlims" : None,
                                                                        "ylims" : None,
                                                                        "bw_method" : None,
                                                                        "barcode_weight" : None,
                                                                        "norm_method" : "sum",
                                                                        "resolution" : 100,
                                                                        "parallel" : True}
                                                }
                                        }


        # Dim reductions parameters
        parameters["Dim_reductions"] = {"morphoframe_filepath" : 0,
                                        "morphoframe_name" : "all_br_bootstrap",
                                        "vectors_to_reduce" : 'pi',
                                        "filter_pixels" : False,
                                        "normalize" : True,
                                        "save_data" : True,
                                        "save_folderpath" : save_folderpath + "/dim_reduced",
                                        "save_filename" : "tmd_morphoframe_cleaned_bootstrap_vect_dimred",
                                        "dimred_method_parameters" : {#"pca" : {"n_components" : 10,
                                        #                                     "svd_solver" : True,
                                        #                                     "pca_version" : 'normal'
                                        #                                     },                           
                                                                "umap" : {"n_components" : 3,
                                                                        "n_neighbors" : 20,
                                                                        "min_dist" : 0.1,
                                                                        "spread" : 3.0,
                                                                        "random_state" : 10,
                                                                        "metric" : "manhattan",
                                                                        "densmap" : False,
                                                                        }
                                                                }
                                        }


        # Saved reduced parameters
        parameters["Save_reduced"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : 'all_br_bootstrap',
                                "conditions_to_save" : ["Region",
                                                        "Model",
                                                        "Sex"],
                                "dimred_method" : "umap",
                                "coordinate_axisnames" : "umap_dim_",
                                "save_folderpath" : save_folderpath + "/dim_reduced",
                                "save_filename" : "tmd_morphoframe_cleaned_bootstrap_vect_dimred",
                                }


        # Plotting parameters
        parameters["Plotting"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : "all_br_bootstrap",
                                "conditions" : ["Region", "Model", "Sex"],
                                "reduced_vectors_name" : "umap",
                                "axis_labels" : ['umap_1', 'umap_2', 'umap_3'],
                                "title" : "umap of pi of tmd microglia",
                                "size" : 3,
                                "colors" : [],
                                'amount' : 0.1,    
                                "save_data" : True,
                                "save_folderpath" : save_folderpath + "/plot",
                                "save_filename" : "tmd_morphoframe_cleaned_bootstrap_vect_dimred_plot",
                                }
        return parameters