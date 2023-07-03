"""
Contains all the default parameters
"""

defaults = {}

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