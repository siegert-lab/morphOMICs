save_folderpath = "br_experiments/results"
brain_region = "FC"

def parameters_umap_plot(save_folderpath = save_folderpath,
                                    brain_region = brain_region,
                                    morphoframe_suffix = "_bootstrap"
                                    ):  
    
    morphoframe_name = brain_region + morphoframe_suffix

    parameters = {}      
    parameters["Parameters_ID"] = brain_region
    parameters["Protocols"] = [ "Input",
                                "Clean_frame",
                                "Bootstrap",
                                "Vectorizations",
                                "Dim_reductions",
                                "Save_reduced",
                                "Plotting",
                                ]
    # Dim reductions parameters
    parameters["Dim_reductions"] = {"morphoframe_filepath" : 0,
                                    "morphoframe_name" : morphoframe_name,
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
                                "morphoframe_name" : morphoframe_name,
                                "conditions_to_save" : ["Region",
                                                        "Model",
                                                        "Sex"],
                                "dimred_method" : "umap",
                                "coordinate_axisnames" : "umap_dim",
                                "save_folderpath" : save_folderpath + "/dim_reduced",
                                "save_filename" : "tmd_morphoframe_cleaned_bootstrap_vect_dimred",
                                }


    # Plotting parameters
    parameters["Plotting"] = {"morphoframe_filepath" : 0,
                            "morphoframe_name" : morphoframe_name,
                            "conditions" : ["Region", "Model", "Sex"],
                            "reduced_vectors_name" : "umap",
                            "axis_labels" : ['umap_1', 'umap_2', 'umap_3'],
                            "title" : "umap of pi of tmd microglia",
                            "size" : 3,
                            "colors" : [],
                            'amount' : 0.1,    
                            "save_data" : True,
                            "save_folderpath" : save_folderpath,
                            "save_filename" : "tmd_morphoframe_cleaned_bootstrap_vect_dimred_plot",
                            }