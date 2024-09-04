"""
Contains all the default parameters for mehtods in Pipeline, Vectorizer and Dim_reducer. 
"""

class DefaultParams:
    def __init__(self):
        self.general_vect_params = {'rescale_lims': True,
                                    'xlims': None,
                                    'norm_method': 'id',
                                    'resolution': 100,
        }
        self.vectorizer_params = {'persistence_image': {'ylims': None,
                                                        'bw_method':None,
                                                        'barcode_weight': None,
                                                        },
                                  'curve': self.general_vect_params,
                                  'hist': self.general_vect_params,
                                  'stable_ranks': {'type': 'standard'},
        }
        self.vectorizer_params['persistence_image'].update(self.general_vect_params)
        self.dimreducer_params = {'pca': {"n_components": 20,
                                          "svd_solver": 'full',
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
                                             'n_neighbors': 50,
                                             'lr': 0.01}
        }
        self.general_io_params ={"morphoframe_filepath": False,
                                "morphoframe_name": 'microglia',
                                "save_data": False,
                                "save_folderpath": 'results',
                                "save_filename": 0
        }
        self.pipeline_params = {'Input': {"data_location_filepath": 'data',
                                            "extension": '.swc',
                                            "conditions": ['Region', 'Model', 'Sex', 'Animal'],
                                            "separated_by": 'Region',
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
                                                "bootstrapframe_name": 'bootstraped_microglia',
                                            },
                                'Vectorizations': {"vect_method_parameters": self.vectorizer_params,
                                                   },
                                'Dim_reductions': {"dimred_method_parameters": self.dimreducer_params,
                                                    "vectors_to_reduce": 'pi',
                                                    "filter_pixels": False,
                                                    "normalize": False,
                                                    "standardize": False,
                                                    "save_dimreducer": False,
                                                },
                                'Save_reduced': {"conditions_to_save": ['Region', 'Model', 'Sex'],
                                                "dimred_method": 'pca_umap',
                                                "coordinate_axisnames": ['umap_1', 'umap_2', 'umap_3'],
                                                },
                                'Mapping': {"fitted_dimreducer_filepath": 0,
                                            "dimred_method": False,
                                            "vectors_to_reduce_name": 'pi',
                                            "filter_pixels": False,
                                            "FilteredPixelIndex_filepath": 0,
                                            },
                                'Palantir': {},
                                'Sholl_curves': {},
                                'Morphometrics': {},
                                'Plotting': {"conditions": ['Region', 'Model', 'Sex'],
                                            "reduced_vectors_name": 'pca_umap',
                                            "axis_labels": ['umap_1', 'umap_2', 'umap_3'],
                                            "title": 'UMAP',
                                            "colors": [],
                                            "circle_colors": [],
                                            "size": 5.,
                                            "amount": 0.3,
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
        completed_params = self._get_default_params(type=type, method=method)
        for key, value in defined_params.items():
            completed_params[key] = value       
        return completed_params
    


