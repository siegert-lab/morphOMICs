import numpy as np

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, TruncatedSVD
import umap
from sklearn.manifold import TSNE

from morphomics.utils import save_obj
from morphomics.utils import norm_methods, scipy_metric
from morphomics.default_parameters import defaults

class Embedder(object):
    
    def __init__(self, tmd_vectors, embed_parameters):
        self.tmd_vectors = tmd_vectors
        self.embed_parameters = embed_parameters
        
    ## Private


    ## Public
    def pca(self):
        pca_params = self.embed_parameters["pca"]

        n_components = pca_params["n_components"]
        
        svd_solver = pca_params["svd_solver"]
        if svd_solver:
            svd_solver = "full"
        else:
            svd_solver = "auto"
        
        pca_version = pca_params["pca_version"]
        
        print("Running PCA...")
        if pca_version == "kernel":
            kernel = pca_params["kernel"]
            fit_pca = KernelPCA(n_components = n_components, kernel = kernel, eigen_solver = svd_solver)
        elif pca_version == "truncated":
            fit_pca = TruncatedSVD(n_components = n_components)
        else:
            fit_pca = PCA(n_components = n_components, svd_solver = svd_solver)

        embedded_vectors = fit_pca.fit_transform(X = self.tmd_vectors)

        return fit_pca, embedded_vectors
    


    def umap(self):
        umap_params = self.embed_parameters["umap"]

        print("Running UMAP...")
        fit_umap = umap.UMAP(
            n_neighbors = umap_params["n_neighbors"],
            min_dist = umap_params["min_dist"],
            spread = umap_params["spread"],
            random_state = umap_params["random_state"],
            n_components = umap_params["n_components"],
            metric = umap_params["metric"],
            densmap = umap_params["densmap"],
        )

        embedded_vectors = fit_umap.fit_transform(X = self.tmd_vectors)

        return fit_umap, embedded_vectors
    


    def tsne(self):

        tsne_params = self.embed_parameters["tsne"]

        n_components = tsne_params["n_components"]
        n_neighbors = tsne_params["n_neighbors"]
        lr = tsne_params["lr"]
        
        print("Running t-SNE...")
        fit_tsne = TSNE(n_components = n_components,
                        perplexity = n_neighbors,
                        learning_rate = lr)
        
        embedded_vectors = fit_tsne.fit_transform(X = self.tmd_vectors)

        return fit_tsne, embedded_vectors
