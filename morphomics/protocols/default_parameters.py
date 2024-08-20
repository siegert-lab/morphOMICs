"""
Contains all the default parameters for mehtods in Pipeline, Vectorizer and Dim_reducer. 
"""

class DefaultParams:
    def __init__(self):
        self.vectorizer_params = {}
        self.dimreducer_params = {}
        self.general_params ={"morphoframe_filepath": False,
                                "morphoframe_name": 'microglia',
                                "save_data": False,
                                "save_folderpath": 'results',
                                "save_filename": 0}
        
        self.pipeline_params = {'Input': {"data_location_filepath": 'data',
                                            "extension": '.swc',
                                            "conditions": ['Region', 'Model', 'Sex', 'Animal'],
                                            "separated_by": ['Region'],
                                            "morphoframe_name": 'microglia',
                                            },
                                'Load_data': {},
                                'TMD': {"filtration_function": 'radial_distance',
                                        "exclude_sg_branches": True,
                                        },
                                'Clean_frame': {},
                                'Subsample': {"extendedframe_name": 'subsampled_microglia',
                                                "feature_to_subsample": 'pi',
                                                "main_branches": 'keep',
                                                "k_elements": 0.9,
                                                "n_samples": 20,
                                                "rand_seed": 51,
                                                },
                                'Bootstrap': {"feature_to_bootstrap": ['pi', 'array'],
                                                "bootstrap_conditions": ['Region', 'Model', 'Sex'],
                                                "N_bags": 200,
                                                "n_samples": 20,
                                                "ratio": 0,
                                                "rand_seed": 51,
                                                "bootstrapframe_name": 'bootstraped_microglia'
                                            },
                                'Vectorizations': {"vect_method_parameters": self.vectorizer_params,
                                                   },
                                'Dim_reductions': {"dimred_method_parameters": self.dimreducer_params,
                                                    "vectors_to_reduce": 'pi',
                                                    "filter_pixels": False,
                                                    "normalize": False,
                                                    "standardize": True,}
        }
    
    def complete_with_default_params(self, defined_params, protocol):
        completed_params = self.pipeline_params[protocol].copy()
        completed_params.update(self.general_params.copy())
        for key, value in defined_params.items():
            completed_params[key] = value       
        return completed_params





# define default image parameter values
defaults['image_parameters'] = {}
defaults['image_parameters']["xlims"] = None
defaults['image_parameters']["ylims"] = None
defaults['image_parameters']["norm_method"] = "sum"
defaults['image_parameters']["metric"] = "l1"
defaults['image_parameters']["chunks"] = 10
defaults['image_parameters']["cutoff"] = 5

defaults['UMAP_parameters'] = {}
defaults['UMAP_parameters']["N_dims"] = 10
defaults['UMAP_parameters']["n_neighbors"] = 20
defaults['UMAP_parameters']["min_dist"] = 1.0
defaults['UMAP_parameters']["spread"] = 3.0
defaults['UMAP_parameters']["random_state"] = 10
