import numpy as np

from morphomics.protocols.default_parameters import DefaultParams
from morphomics.persistent_homology import vectorizations
from morphomics.persistent_homology.ph_analysis import get_limits

from morphomics.utils import norm_methods


class Vectorizer(object):
    
    def __init__(self, tmd, vect_parameters):
        """
        Initializes the Vectorizer instance.

        Parameters
        ----------
        tmd (list of np.arrays of pairs): the barcodes of trees
        parameters (dict): contains the parameters for each protocol that would be run
                            vect_parameters = {'vect_method_1 : { parameter_1_1: x_1_1, ..., parameter_1_n: x_1_n},
                                                 ...
                                                'vect_method_m : { parameter_m_1: x_m_1, ..., parameter_m_n: x_m_n}
                                        } 

        Returns
        -------
        An instance of Vectorizer.
        """
        self.tmd = tmd
        self.vect_parameters = vect_parameters
        self.default_params = DefaultParams()

    ## Private
    def _curve_vectorization(self, 
                             curve_params,
                             curve_method):
        '''General method to compute vectorization for curve methods.

        Parameters
        ----------
        curve_method (method): the actual curve like vectorization method, the method has 2 parameters: (t_list, resolution).
                                example: lifespan_curve.
        curve_params (dict): the parameters for the curve vectorization:
                            -rescale_lims (bool): True: adapt the boundaries of the barcode for each barcode
                                                False: choose the widest boundaries that include all barcodes 
                            -xlims (pair of double): the boundaries
                            -resolution (int): number of real values for which the fonction is computed .aka. size of the output vector 
                            -norm_method (str): the method to normalize the vector

        Returns
        -------
        A numpy array of shape (nb barcodes, resolution) i.e. a vector for each barcode. 
        '''
        curve_params = self.default_params.complete_with_default_params(curve_params, "curve", type = 'vectorization')
        rescale_lims = curve_params["rescale_lims"]
        xlims = curve_params["xlims"]
        resolution = curve_params["resolution"]
        norm_method = curve_params["norm_method"]

        # Define the real values where the curve is computed
        if rescale_lims:
            t_list = None
        else:
            if xlims is None or xlims == "None":
                # get the birth and death distance limits for the curve
                _xlims, _ylims = get_limits(self.tmd)
                xlims = [np.min([_xlims[0], _ylims[0]]), np.max([_xlims[1], _ylims[1]])]
            t_list = np.linspace(xlims[0], xlims[1], resolution)
        # Get the curve
        curve_list = self.tmd.apply(lambda ph: curve_method(ph,
                                                            t_list = t_list, 
                                                            resolution = resolution)[0]
        )
        # normalize the curve
        norm_m = norm_methods[norm_method]
        curves = curve_list.apply(lambda curve: curve/norm_m(curve) if len(curve)>0 else np.nan)

        return np.array(list(curves))
    

    def _histogram_vectorization(self, 
                             hist_params,
                             hist_method):
        '''General method to compute vectorization for histogram methods.

        Parameters
        ----------
        hist_method (method): the actual histogram like vectorization method, the method has 2 parameters: (bins, num_bins).
                                example: betti_hist.
        hist_params (dict): the parameters for the histogram vectorization:
                            -rescale_lims (bool): True: adapt the boundaries of the barcode for each barcode
                                                False: choose the widest boundaries that include all barcodes 
                            -xlims (pair of double): the boundaries
                            -num_bins (int): number of sub intervals between boudaries .aka. size of the output vector 
                            -norm_method (str): the method to normalize the vector

        Returns
        -------
        A numpy array of shape (nb barcodes, resolution) i.e. a vector for each barcode. 
        '''
        hist_params = self.default_params.complete_with_default_params(hist_params, "hist", type = 'vectorization')
        rescale_lims = hist_params["rescale_lims"]
        xlims = hist_params["xlims"]
        num_bins = hist_params["resolution"]
        norm_method = hist_params["norm_method"]

        # Define the sub intervals of the histogram
        if rescale_lims:
            bins = None
        else:
            if xlims is None or xlims == "None":
                # get the birth and death distance limits for the curve
                _xlims, _ylims = get_limits(self.tmd)
                xlims = [np.min([_xlims[0], _ylims[0]]), np.max([_xlims[1], _ylims[1]])]
            bins = vectorizations._subintervals(xlims = xlims, num_bins = 1000)
        # Get the curve
        hist_list = self.tmd.apply(lambda ph: hist_method(ph,
                                                        bins = bins, 
                                                        num_bins = num_bins)[0]
        )
        # normalize the curve
        norm_m = norm_methods[norm_method]
        histograms = hist_list.apply(lambda hist: hist/norm_m(hist) if len(hist)>0 else np.nan)

        return np.array(list(histograms))
    

    ## Public
    def persistence_image(self):
        '''This function takes information about barcodes, calculates persistence images based on specified
        parameters, and returns an array of images.
        
        Parameters
        ----------
            The 'rescale_lims' is a boolean. 
                True: adapt the boundaries of the barcode for each barcode
                False: choose the widest boundaries that include all barcodes 
            The `xlims` parameter used to specify the
        birth and death distance limits for the persistence images. If `xlims` is not provided as an
        argument when calling the function, it will default to the birth and death distance limits
        calculated from
        ylims
            The `ylims` parameter is used to specify the
        limits for the y-axis in the persistence images. If `ylims` is not provided as an argument when
        calling the function, it will default to `None` and then be set based
        bw_method
            The `bw_method` parameter is used to specify the
        bandwidth method for kernel density estimation when generating persistence images. It controls the
        smoothness of the resulting images by adjusting the bandwidth of the kernel used in the estimation
        process. Different bandwidth methods can result
        norm_method, optional
            The `norm_method` parameter specifies the method
        used for normalizing the persistence images. The default value is set to "sum", which means that the
        images will be normalized by dividing each pixel value by the sum of all pixel values in the image
        barcode_weight
            The `barcode_weight` parameter is used to specify
        weights for each barcode in the calculation of persistence images. If `barcode_weight` is provided,
        it will be used as weights for the corresponding barcode during the calculation. 
           The 'resolution' parameter is an integer that defines the number of pixels in a row and in a column of a persistence image.
        
        Returns
        -------
            The function returns a NumPy array of persistence images.
        

        '''
        pi_params = self.vect_parameters["persistence_image"]
        pi_params = self.default_params.complete_with_default_params(pi_params, "persistence_image", type = 'vectorization')

        rescale_lims = pi_params["rescale_lims"]
        xlims=pi_params["xlims"]
        ylims=pi_params["ylims"]
        bw_method=pi_params["bw_method"]
        if bw_method == "None":
            bw_method = None
        barcode_weight=pi_params["barcode_weight"]
        if barcode_weight == "None":
            barcode_weight = None
        norm_method=pi_params["norm_method"]
        resolution=pi_params["resolution"]
        flatten = True

        print("Computing persistence images...")
        
        if rescale_lims:
            xlims, ylims = None, None
        else:
            # get the birth and death distance limits for the persistence images
            ph_list = list(self.tmd)
            _xlims, _ylims = vectorizations.get_limits(ph_list)
            if xlims is None or xlims == "None":
                xlims = _xlims
            if ylims is None or ylims == "None":
                ylims = _ylims
        
        pi_list = self.tmd.apply(lambda ph: vectorizations.persistence_image(ph,
                                                                            xlim = xlims,
                                                                            ylim = ylims,
                                                                            bw_method = bw_method,
                                                                            weights = barcode_weight,
                                                                            resolution = resolution,
                                                                            )
        )
        if flatten:
            pi_list = pi_list.apply(lambda pi: pi.flatten())

        norm_m = norm_methods[norm_method]
        pis = pi_list.apply(lambda pi: pi/norm_m(pi) if len(pi)>0 else np.nan)
    
        print("pi done! \n")
        print(np.array(pis).shape)
        return np.array(list(pis))


    def betti_curve(self):
        ''' Computes the betti curve of each barcode in self.tmd.

        Parameters
        ----------
        betti_params (dict): the parameters for the betti curve vectorization:
                            -rescale_lims (bool): True: adapt the boundaries of the barcode for each barcode
                                                False: choose the widest boundaries that include all barcodes 
                            -xlims (pair of double): the boundaries
                            -resolution (int): number of sub intervals between boudaries .aka. size of the output vector 
                            -norm_method (str): the method to normalize the vector

        Returns
        -------
        A numpy array of shape (nb barcodes, resolution) i.e. a vector for each barcode. 
        '''
        betti_params = self.vect_parameters["betti_curve"]
        print("Computing betti curves...")
        betti_curves = self._curve_vectorization(curve_params = betti_params,
                                                curve_method = vectorizations.betti_curve)
        print("bc done! \n")
        return betti_curves


    def life_entropy_curve(self):
        ''' Computes the life entropy curve of each barcode in self.tmd.

        Parameters
        ----------
        entropy_params (dict): the parameters for the life entropy curve vectorization:
                            -rescale_lims (bool): True: adapt the boundaries of the barcode for each barcode
                                                False: choose the widest boundaries that include all barcodes 
                            -xlims (pair of double): the boundaries
                            -resolution (int): number of sub intervals between boudaries .aka. size of the output vector 
                            -norm_method (str): the method to normalize the vector

        Returns
        -------
        A numpy array of shape (nb barcodes, resolution) i.e. a vector for each barcode. 
        '''
        entropy_params = self.vect_parameters["life_entropy_curve"]
        print("Computing life entropy curves...")
        life_entropy_curves = self._curve_vectorization(curve_params = entropy_params,
                                                        curve_method = vectorizations.life_entropy_curve)
        print("lec done! \n")
        return life_entropy_curves


    def lifespan_curve(self):
        ''' Computes the life span curve of each barcode in self.tmd.

        Parameters
        ----------
        lifespan_params (dict): the parameters for the life span curve vectorization:
                            -rescale_lims (bool): True: adapt the boundaries of the barcode for each barcode
                                                False: choose the widest boundaries that include all barcodes 
                            -xlims (pair of double): the boundaries
                            -resolution (int): number of sub intervals between boudaries .aka. size of the output vector 
                            -norm_method (str): the method to normalize the vector

        Returns
        -------
        A numpy array of shape (nb barcodes, resolution) i.e. a vector for each barcode. 
        '''
        lifespan_params = self.vect_parameters["lifespan_curve"]
        print("Computing lifespan curves...")
        lifespan_curves = self._curve_vectorization(curve_params = lifespan_params,
                                                    curve_method = vectorizations.lifespan_curve)
        print("lsc done! \n")
        return lifespan_curves


    def stable_ranks(self):
        stable_ranks_params = self.vect_parameters["stable_ranks"]
        # Type should be: 'standard', 'abs' or 'positiv'.
        stable_ranks_params = self.default_params.complete_with_default_params(stable_ranks_params, "stable_ranks", type = 'vectorization')
        type = stable_ranks_params['type']
        print("Computing stable ranks...")
        stable_r = self.tmd.apply(lambda ph: vectorizations.stable_ranks(ph, type = type))
        # Determine the maximum length of vectors
        max_len = stable_r.apply(len).max()

        # Pad each vector with zeros
        stable_r = stable_r.apply(lambda x: np.pad(x, (0, max_len - len(x)), mode='constant'))

        print("sr done! \n")
        return np.array(list(stable_r))


    def betti_hist(self):
        ''' Computes the betti histogram of each barcode in self.tmd.

        Parameters
        ----------
        betti_hist_params (dict): the parameters for the betti histogram vectorization:
                            -bins (list, pair): The list of subintervals where betti number is computed.
                            -num_bins (int): The number of bins if bins is not defined.
                            -norm_method (str): The method to normalize the vector

        Returns
        -------
        A numpy array of shape (nb barcodes, len(bins)) i.e. a vector for each barcode. 
        '''
        betti_hist_params = self.vect_parameters["betti_hist"]
        print("Computing betti histograms...")
        betti_hists = self._histogram_vectorization(hist_params = betti_hist_params,
                                                    hist_method = vectorizations.betti_hist)
        print("bh done! \n")
        return np.array(list(betti_hists))


    def lifespan_hist(self):
        ''' Computes the lifespan histogram of each barcode in self.tmd.

        Parameters
        ----------
        lifespan_hist_params (dict): the parameters for the lifespan histogram vectorization:
                            -bins (list, pair): The list of subintervals where lifespan is computed.
                            -num_bins (int): The number of bins if bins is not defined.
                            -norm_method (str): The method to normalize the vector

        Returns
        -------
        A numpy array of shape (nb barcodes, len(bins)) i.e. a vector for each barcode. 
        '''
        lifespan_hist_params = self.vect_parameters["lifespan_hist"]
        print("Computing lifespan histograms...")
        lifespan_hists = self._histogram_vectorization(hist_params = lifespan_hist_params,
                                                        hist_method = vectorizations.lifespan_hist)
        print("lh done! \n")
        return np.array(list(lifespan_hists))

