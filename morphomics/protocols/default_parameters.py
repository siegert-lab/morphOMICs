import warnings
import torch.nn as nn
"""
Contains all the default parameters for mehtods in Pipeline, Vectorizer and Dim_reducer. 
"""

class DefaultParams:
    def __init__(self):
        self.general_vect_params = {'rescale_lims': True,
                                    'xlims': None,
                                    'resolution': 100,
                                    'norm_method': 'id',
        }
        self.vectorizer_params = {'persistence_image': {'ylims': None,
                                                        'method': 'kde',
                                                        'std_isotropic': 0.1,
                                                        'bw_method':None,
                                                        'barcode_weight': None,
                                                        },
                                  'curve': self.general_vect_params,
                                  'lifespan_curve': self.general_vect_params,
                                  'betti_curve': self.general_vect_params,
                                  'life_entropy_curve': self.general_vect_params,
                                  'hist': self.general_vect_params,
                                  'betti_hist': self.general_vect_params,
                                  'lifespan_hist': self.general_vect_params,
                                  'stable_ranks': {'type': 'neg', 'density': False, 'bars_prob': False, 'resolution': 1000},
        }
        self.vectorizer_params['persistence_image'].update(self.general_vect_params)
        self.general_dl_params = {'batch_layer_norm': False,
                                'optimizer': 'cocob',
                                'learning_rate': None,
                                'momentum': 0.9,
                                'scheduler': False,
                                'nb_epochs': 100,
                                'batch_size': 32,}
        self.dimreducer_params = {'pca': {"n_components": 20,
                                          "svd_solver": False,
                                          "pca_version": 'standard',
                                            },
                                    'umap': {"n_neighbors": 50,
                                             "n_components": 2,
                                             "min_dist": 0.1,
                                             "spread": 3.,
                                             "random_state": 51,
                                             "metric": "manhattan",
                                             "densmap": False,
                                             },
                                    'tsne': {'n_components': 2,
                                             'perplexity': 50,
                                             'lr': "auto"
                                             },
                                    'vae': {'n_components': 2,      
                                            'nn_layers': [64, 32, 16, 8], 
                                            'activation_layer': nn.SELU,
                                            'kl_factor_function' : None,
                                            },
                                    'vaecnn': {'n_components': 2,      
                                            'nn_layers': [8, 16, 32], 
                                            'kl_factor_function' : None, 
                                            }
        }
        self.dimreducer_params['vae'].update(self.general_dl_params)
        self.dimreducer_params['vaecnn'].update(self.general_dl_params)

        self.general_io_params ={"morphoframe_filepath": False,
                                "morphoframe_name": 'microglia',
                                "save_data": False,
                                "save_folderpath": 'results',
                                "save_filename": 0
        }
        self.pipeline_params = {'Input': {"data_location_filepath": 'data',
                                            "extension": '.swc',
                                            "conditions": ['Region', 'Model', 'Sex', 'Animal'],
                                            "separated_by": None,
                                            },
                                'TMD': {"filtration_function": 'radial_distance',
                                        },
                                'Clean_frame': {"combine_conditions": [],
                                                "restrict_conditions": []
                                                },
                                'Filter_frame': {"barcode_size_cutoff": 5,
                                                 "features_to_filter":  { "nb_trunks" : { 'min': 0, 'max': 10, 'type': 'abs' },
                                                                        "max_length_bar" : { 'min': 0, 'max': 150, 'type': 'abs' },
                                                                        "nb_bars" : { 'min': 5, 'max': 250, 'type': 'abs' } }
                                                 },
                                'Filter_morpho': {"barlength_cutoff": [],
                                                "exclude_sg_branches": True,
                                                  },
                                'Subsample': {"extendedframe_name": 'subsampled_microglia',
                                                "feature_to_subsample": 'barcodes',
                                                "n_samples": 20,
                                                "rand_seed": 51,
                                                
                                                "main_branches": 'keep',
                                                "k_elements": 0.9,
                                                "type": "keep",
                                                "nb_sections": 1,
                                                },
                                'Bootstrap': {"feature_to_bootstrap": ['pi', 'array'],
                                                "bootstrap_conditions": ['Region', 'Model', 'Sex'],
                                                "numeric_condition": False,
                                                "numeric_condition_std": 1.5,
                                                "N_bags": 200,
                                                "n_samples": 20,
                                                "ratio": 0,
                                                "replacement": True,
                                                "rand_seed": None,
                                                "bootstrapframe_name": 'bootstraped_microglia',
                                            },
                                'Vectorizations': {"vect_method_parameters": self.vectorizer_params,
                                                   "barcode_column": 'barcodes'
                                                   },
                                'Dim_reductions': {"dimred_method_parameters": self.dimreducer_params,
                                                    "vectors_to_reduce": 'pi',
                                                    "filter_pixels": False,
                                                    "normalize": False,
                                                    "standardize": False,
                                                    "save_dimreducer": False,
                                                    "FilteredPixelIndex_filepath": False,
                                                    "pixel_std_cutoff": 1e-5
                                                },
                                'Save_reduced': {"conditions_to_save": ['Region', 'Model', 'Sex'],
                                                "dimred_method": 'pca_umap',
                                                "coordinate_axisnames": ['umap_1', 'umap_2', 'umap_3'],
                                                },
                                'Log_results': {"checks":[]},
                                'Mapping': {"fitted_dimreducer_filepath": 0,
                                            "dimred_method": False,
                                            "vectors_to_reduce_name": 'pi',
                                            "filter_pixels": False,
                                            "FilteredPixelIndex_filepath": False,
                                            "pixel_std_cutoff": 1e-4,
                                            },
                                'Palantir': {},
                                'Sholl_curves': {},
                                'Morphometrics': {"Lmeasure_functions": None,
                                                  "concatenate": False,
                                                  "histogram": False,
                                                  "bins": 100,
                                                  "tmp_folder": "results"},
                                'Plotting': {"conditions": ['Region', 'Model', 'Sex'],
                                            "reduced_vectors_name": 'pca_umap',
                                            "axis_labels": ['umap_1', 'umap_2', 'umap_3'],
                                            "title": 'UMAP',
                                            "colors": [],
                                            "circle_colors": [],
                                            "size": 5.,
                                            "amount": 0.3,
                                            "show": True,
                                            },
                                'Save_parameters': {"parameters_to_save": {'TMD': ['filtration_function'],
                                                                            'Bootstrap': ['N_bags', 'n_samples', 'ratio'],
                                                                            'Vectorizations': [],
                                                                            'Dim_reductions': ['normalize', 'standardize'],
                                                                            }
                                                    },
        }
    
    def _get_default_params(self, type, method):
        if type == 'protocol':
            completed_params = self.pipeline_params[method].copy()
            completed_params.update(self.general_io_params.copy())
        elif type == 'vectorization':
            completed_params = self.vectorizer_params[method].copy()
        elif type == 'dim_reduction':
            completed_params = self.dimreducer_params[method].copy()
        return completed_params
    

    def complete_with_default_params(self, defined_params, method, type = 'protocol'):
        # Complete the input parameters with the default one.
        completed_params = self._get_default_params(type=type, method=method)
        for key, value in defined_params.items():
            completed_params[key] = value
        
        return completed_params
    
    def check_params(self, defined_params, method, type = 'protocol'):
        # Check if all input parameters correspond to a meaningful parameter.
        param_names = self._get_default_params(type=type, method=method).keys()
        not_params_list = []
        for key in defined_params.keys():
            if key not in param_names:
                not_params_list.append(key)
        if len(not_params_list) != 0:
            # Create a warning if the list is not empty
            warnings.warn(f"The following parameter names are not correct: {not_params_list}", UserWarning)
            raise ValueError(f"Error: Invalid elements found in the list: {not_params_list}")
        else:
            print("All parameter names are correct.")        





