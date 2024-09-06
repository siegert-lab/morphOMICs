from morphomics import pipeline, utils

def run_pipeline(path_to_data,
             pid, mf_name,
             ext_name,
             bf_name,
             fmf_name ,
             clean_mf,
             prot_to_clean, 
             save_folderpath,
             param_id,
             protocol_list,
             save_list,
             # Input
             conditions = ['Region', 'Model', 'Sex', 'Animal'],
             extension = "_corrected.swc",
             # TMD
             filtration = "radial_distance",
             # Subsample
                feature_sample = 'tree',
                # Vect
                vect_method_parameters = {'persistence_image' : {"rescale_lims" : True,
                        "xlims" : [-10, 160],
                        "ylims" : [-10, 160],
                        "bw_method" : None,
                        "barcode_weight" : None,
                        "norm_method" : "id",
                        "resolution" : 80,
                        }},
                # BT
                bt_cond = ['Model'],
                ratio = 0,
                N_bags = 100,
                n_samples = 30,
                # UMAP
                filter_pixels = False,
                pixel_std_cutoff = 0.0001,
                dimred_method_parameters = {"pca" : {"n_components" : 50,
                                                                        "svd_solver" : False,
                                                                        "pca_version" : 'normal'
                                                                        },                           
                                                                "umap" : {"n_components" : 3,
                                                                        "n_neighbors" : 50,
                                                                        "min_dist" : 0.05,
                                                                        "spread" : 3.0,
                                                                        "random_state" : 10,
                                                                        "metric" : "manhattan",
                                                                        "densmap" : False,
                                                                        }
                                                                },
                norm = False,
                stand = False,
                #plot
                plot_title = 'UMAP',
                colors = [],
                circle_colors = None):
        
        parameters = {}
        parameters["Parameters_ID"] = pid
        parameters["Protocols"] = protocol_list

        my_pipeline = pipeline.Pipeline(parameters = parameters, 
                                        Parameters_ID = parameters["Parameters_ID"])
        
        # Input parameters
        my_pipeline.parameters["Input"] = {"data_location_filepath" : path_to_data,
                        "extension" : extension,
                        "conditions" : conditions,  
                        "separated_by" : conditions[0],
                        "morphoframe_name" : mf_name,
                        "save_data" : False,
                        "save_folderpath" : save_folderpath + "/tmd",
                        "save_filename" : mf_name
                        }

        # Clean_frame parameters
        my_pipeline.parameters["Clean_frame"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : mf_name,
                                "barcode_size_cutoff" : 5,
                                "barlength_cutoff" : [],
                                "combine_conditions" : [],
                                "restrict_conditions" : [],
                                "save_data" : False,
                                "save_folderpath" : save_folderpath + "/tmd",
                                "save_filename" : mf_name + "_cleaned",
                                }


        my_pipeline.parameters['Load_data'] = {"filepath_to_data" : path_to_data,
                                                "morphoframe_name" : mf_name}
        
        my_pipeline.parameters['TMD'] = {"morphoframe_filepath" : 0,
                                         "morphoframe_name" : mf_name,
                                         "filtration_function" : filtration,
                                         "exclude_sg_branches" : True,
                                        "save_data" : False,
                                        "save_folderpath" : save_folderpath + "/tmd",
                                        "save_filename" : mf_name
                                        }
        
        my_pipeline.parameters['Subsample'] = {"morphoframe_filepath" : 0,
                                               "morphoframe_name" : mf_name,
                                               "extendedframe_name" : ext_name,
                                                "feature_to_subsample": feature_sample,
                                                "n_samples" : n_samples,
                                                "rand_seed" : 51,
                                                "save_data" : False,
                                                "save_folderpath" : save_folderpath + "/sample",
                                                "save_filename" : mf_name + "_cleaned",
                                                }

        # Vectorization parameters
        my_pipeline.parameters["Vectorizations"] = {"morphoframe_filepath" : 0,
                                                "morphoframe_name" : mf_name,
                                                "save_data" : False,
                                                "save_folderpath" : save_folderpath + "/vectorized" + param_id,
                                                "save_filename" : mf_name,
                                                "vect_method_parameters" :
                                                vect_method_parameters
                                                }
        
        vect_method_name = utils.vectorization_codenames[list(vect_method_parameters.keys())[0]]
        # Bootstrap parameters
        my_pipeline.parameters["Bootstrap"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : mf_name,
                                "feature_to_bootstrap" : [vect_method_name, "array"],         
                                "bootstrap_conditions" : bt_cond,
                                "rand_seed" : 34151,
                                "ratio" : ratio,
                                "N_bags" : N_bags,
                                "n_samples" : n_samples,
                                "bootstrapframe_name" : bf_name,
                                "save_data" : False,
                                "save_folderpath" : save_folderpath + "/bootstrap" + param_id,
                                "save_filename" : bf_name,
                                }

        # Dim reductions parameters
        my_pipeline.parameters["Dim_reductions"] = {"morphoframe_filepath" : 0,
                                        "morphoframe_name" : fmf_name,
                                        "vectors_to_reduce" : vect_method_name,
                                        "filter_pixels" : filter_pixels,
                                        "FilteredPixelIndex_filepath": False,
                                        "pixel_std_cutoff": pixel_std_cutoff,
                                        "normalize" : norm,
                                        "standardize" : stand,
                                        "save_data" : False,
                                        "save_folderpath" : save_folderpath + "/dim_reduced" + param_id,
                                        "save_filename" : fmf_name,
                                        "dimred_method_parameters" : dimred_method_parameters
                                        }
        dimred_method_names = list(dimred_method_parameters.keys())
        dimred_method_name = '_'.join(dimred_method_names)

        # Saved reduced parameters
        my_pipeline.parameters["Save_reduced"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : fmf_name,
                                "conditions_to_save" : bt_cond,
                                "dimred_method" : dimred_method_name,
                                "coordinate_axisnames" : "umap_dim_",
                                "save_folderpath" : save_folderpath + "/dim_reduced" + param_id,
                                "save_filename" : fmf_name,
                                }
        
        # Plotting parameters
        my_pipeline.parameters["Plotting"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : fmf_name,
                                "conditions" : bt_cond,
                                "reduced_vectors_name" : dimred_method_name,
                                "axis_labels" : ['umap_dim_1', 'umap_dim_2', 'umap_dim_3'],
                                "title" : plot_title,
                                "size" : 5,
                                "colors" : colors,
                                "circle_colors" : circle_colors,
                                'amount' : 0.1,    
                                "save_data" : False,
                                "save_folderpath" : save_folderpath + "/plot",
                                "save_filename" : fmf_name,
                                }
        
        for prot, save_data in zip(my_pipeline.parameters['Protocols'], save_list):
                my_pipeline.parameters[prot]['save_data'] = save_data
                perform_this = getattr(my_pipeline, prot)
                perform_this()
                if prot == prot_to_clean and clean_mf:
                        original_mf = my_pipeline.morphoframe[mf_name]
                        my_pipeline.morphoframe[mf_name] = clean_mf(original_mf)

        return my_pipeline