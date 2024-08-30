import numpy as np
from morphomics.protocols.default_parameters import DefaultParams
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, TruncatedSVD
import umap
from sklearn.manifold import TSNE


class DimReducer(object):

    def __init__(self, tmd_vectors, dimred_parameters):
        """
        Initializes the Vectorizer instance.

        Parameters
        ----------
        tmd_vectors (np.array): array containing vectors, one vector per sample in row.
        dimred_parameters (dict): contains the parameters for each dim reduction techniques that would be run in sequence.
                            dimred_parameters = {'dimred_method_1 : { parameter_1_1: x_1_1, ..., parameter_1_n: x_1_n},
                                                ...
                                                'dimred_method_m : { parameter_m_1: x_m_1, ..., parameter_m_n: x_m_n}
                                                } 

        Returns
        -------
        An instance of DimReducer.
        """
        self.tmd_vectors = tmd_vectors
        self.dimred_parameters = dimred_parameters
        self.default_params = DefaultParams()

    ## Private


    ## Public
    def pca(self):
        ''' Dim reduction of tmd vectors with pca.

        Parameters
        ----------
        pca_params (dict): the parameters for the pca:
                            -n_components (int): Number of components to keep. if n_components is not set all components are kept.
                            -svd_solver (bool): Computes the exact SVD, otherwise the solver is selected by a default ‘auto’ policy defined by scikit learn.
                            -pca_version (str): Select the type of pca.
                            -kernel (str): Kernel used for PCA.

        Returns
        -------
        fit_pca (PCA): The fitted instance.
        reduced_vectors (np.array): The dim reduced vectors.
        '''
        pca_params = self.dimred_parameters["pca"]
        pca_params = self.default_params.complete_with_default_params(self, pca_params, 'pca', type = 'dim_reduction')

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

        reduced_vectors = fit_pca.fit_transform(X = self.tmd_vectors)

        return fit_pca, reduced_vectors
    


    def umap(self):
        ''' Dim reduction of tmd vectors with umap.

        Parameters
        ----------
        n_neighbors (int): The size of local neighborhood (in terms of number of neighboring
            sample points) used for manifold approximation. 
            Larger values result in more global views of the manifold, 
            while smaller values result in more local data being preserved. 
            In general values should be in the range 2 to 100.
        n_components (int): The dimension of the space to embed into. This defaults to 2 to provide easy visualization, 
            but can reasonably be set to any integer value in the range 2 to 100.
        metric (str): The metric to use to compute distances in high dimensional space. 
            If a string is passed it must match a valid predefined metric. 
            If a general metric is required a function that takes two 1d arrays and returns a float can be provided. 
            For performance purposes it is required that this be a numba jit’d function.
        min_dist (float): The effective minimum distance between embedded points. 
            Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, 
            while larger values will result on a more even dispersal of points. 
            The value should be set relative to the spread value, 
            which determines the scale at which embedded points will be spread out.
        spread (float): The effective scale of embedded points. 
            In combination with min_dist this determines how clustered/clumped the embedded points are.
        random_state (int): If int, random_state is the seed used by the random number generator; 
            If RandomState instance, random_state is the random number generator; 
            If None, the random number generator is the RandomState instance used by np.random.
        densmap (bool): Specifies whether the density-augmented objective of densMAP should be used for optimization. 
            Turning on this option generates an embedding where the local densities are encouraged to be correlated with those in the original space. 
            Parameters below with the prefix ‘dens’ further control the behavior of this extension.

        Returns
        -------
        fit_umap (UMAP): The fitted instance.
        reduced_vectors (np.array): The dim reduced vectors.
        '''
        umap_params = self.dimred_parameters["umap"]
        umap_params = self.default_params.complete_with_default_params(self, umap_params, 'umap', type = 'dim_reduction')

        print("Running UMAP...")
        fit_umap = umap.UMAP(
            n_neighbors = umap_params["n_neighbors"],
            n_components = umap_params["n_components"],
            min_dist = umap_params["min_dist"],
            spread = umap_params["spread"],
            random_state = umap_params["random_state"],
            metric = umap_params["metric"],
            densmap = umap_params["densmap"],
        )

        reduced_vectors = fit_umap.fit_transform(X = self.tmd_vectors)

        return fit_umap, reduced_vectors
    


    def tsne(self):
        ''' Dim reduction of tmd vectors with umap.

        Parameters
        ----------
        n_components (int): Dimension of the embedded space.
        perplexity (float): The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. 
            Larger datasets usually require a larger perplexity. 
            Consider selcting a value between 5 and 50. 
            The choice is not extremely critical since t-SNE is quite insensitive to this parameter.
        learning_rate (float): The learning rate can be a critical parameter. 
            It should be between 100 and 1000. 
            If the cost function increases during initial optimization, 
            the early exaggeration factor or the learning rate might be too high. 
            If the cost function gets stuck in a bad local minimum increasing the learning rate helps sometimes.
        
        Returns
        -------
        fit_tsne (TSNE): The fitted instance.
        reduced_vectors (np.array): The dim reduced vectors.
        '''
        tsne_params = self.dimred_parameters["tsne"]
        tsne_params = self.default_params.complete_with_default_params(self, tsne_params, 'tsne', type = 'dim_reduction')

        n_components = tsne_params["n_components"]
        n_neighbors = tsne_params["n_neighbors"]
        lr = tsne_params["lr"]
        
        print("Running t-SNE...")
        fit_tsne = TSNE(n_components = n_components,
                        perplexity = n_neighbors,
                        learning_rate = lr)
        
        reduced_vectors = fit_tsne.fit_transform(X = self.tmd_vectors)

        return fit_tsne, reduced_vectors



    def autoencoder(self):
        return

 