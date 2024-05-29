import numpy as np

import umap

from morphomics.utils import save_obj
from morphomics.utils import norm_methods, scipy_metric
from morphomics.default_parameters import defaults

class Embedder(object):
    
    def __init__(self, embed_methods, embed_parameters):
        self.embed_methods = embed_methods
        self.embed_parameters = embed_parameters
        
    ## Private


    ## Public