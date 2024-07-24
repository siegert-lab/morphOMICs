from morphomics import protocols, utils

def pipeline(path_to_data,
             pid, mf_name,
             bf_name,
             fmf_name ,
             clean_mf,
             prot_to_clean, 
             save_folderpath,
             param_id,
             protocol_list,
             save_list,
             # Input
             conditions,
             extension = "_corrected.swc",
          #vect
        vect_method_parameters = {'persistence_image' : {"rescale_lims" : False,
                        "xlims" : [-10, 160],
                        "ylims" : [-10, 160],
                        "bw_method" : None,
                        "barcode_weight" : None,
                        "norm_method" : "one",
                        "resolution" : 80,
                        "parallel" : False}},
                #bt
                bt_cond = ['Model'],
                ratio = 0,
                N_bags = 100,
                n_samples = 30,
                #umap
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

        protocol = protocols.Protocols(parameters = parameters, 
                                        Parameters_ID = parameters["Parameters_ID"])
        
        # Input parameters
        protocol.parameters["Input"] = {"data_location_filepath" : path_to_data,
                        "extension" : extension,
                        "conditions" : conditions,  
                        "separated_by" : conditions[0],
                        "filtration_function" : "radial_distances",
                        "morphoframe_name" : mf_name,
                        "save_data" : False,
                        "save_folderpath" : save_folderpath + "/tmd",
                        "save_filename" : mf_name
                        }

        # Clean_frame parameters
        protocol.parameters["Clean_frame"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : mf_name,
                                "barcode_size_cutoff" : 5,
                                "barlength_cutoff" : [],
                                "combine_conditions" : [],
                                "restrict_conditions" : [],
                                "save_data" : False,
                                "save_folderpath" : save_folderpath + "/tmd",
                                "save_filename" : mf_name + "_cleaned",
                                }


        protocol.parameters['Load_data'] = {"filepath_to_data" : path_to_data,
                                                "morphoframe_name" : mf_name}
        

        # Vectorization parameters
        protocol.parameters["Vectorizations"] = {"morphoframe_filepath" : 0,
                                        "morphoframe_name" : mf_name,
                                        "save_data" : False,
                                        "save_folderpath" : save_folderpath + "/vectorized" + param_id,
                                        "save_filename" : mf_name,
                                        "vect_method_parameters" :
                                             vect_method_parameters
                                        }
        
        # Bootstrap parameters
        protocol.parameters["Bootstrap"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : mf_name,
                                "feature_to_bootstrap" : ["pi", "array"],         
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
        protocol.parameters["Dim_reductions"] = {"morphoframe_filepath" : 0,
                                        "morphoframe_name" : fmf_name,
                                        "vectors_to_reduce" : 'pi',
                                        "filter_pixels" : False,
                                        "normalize" : norm,
                                        "standardize" : stand,
                                        "save_data" : False,
                                        "save_folderpath" : save_folderpath + "/dim_reduced" + param_id,
                                        "save_filename" : bf_name,
                                        "dimred_method_parameters" : dimred_method_parameters
                                        }
        
        # Saved reduced parameters
        protocol.parameters["Save_reduced"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : fmf_name,
                                "conditions_to_save" : bt_cond,
                                "dimred_method" : "pca_umap",
                                "coordinate_axisnames" : "umap_dim_",
                                "save_folderpath" : save_folderpath + "/dim_reduced" + param_id,
                                "save_filename" : fmf_name,
                                }
        
        # Plotting parameters
        protocol.parameters["Plotting"] = {"morphoframe_filepath" : 0,
                                "morphoframe_name" : fmf_name,
                                "conditions" : bt_cond,
                                "reduced_vectors_name" : "pca_umap",
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
        
        for prot, save_data in zip(protocol.parameters['Protocols'], save_list):
                protocol.parameters[prot]['save_data'] = save_data
                perform_this = getattr(protocol, prot)
                perform_this()
                if prot == prot_to_clean and clean_mf:
                        original_mf = protocol.morphoframe[mf_name]
                        protocol.morphoframe[mf_name] = clean_mf(original_mf)

        return protocol