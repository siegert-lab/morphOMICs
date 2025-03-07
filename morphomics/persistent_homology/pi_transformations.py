import numpy as np
from morphomics.persistent_homology.vectorizations import persistence_image

def get_image_add_data(Z1, Z2, normalized=True):
    """Get the sum of two images from the gaussian kernel plotting function."""
    if normalized:
        Z1 = Z1 / Z1.max()
        Z2 = Z2 / Z2.max()
    return Z1 + Z2

def get_image_diff_data(Z1, Z2, normalized=True):
    """Get the diff of two images from the gaussian kernel plotting function."""
    if normalized:
        Z1 = Z1 / Z1.max()
        Z2 = Z2 / Z2.max()
    return Z1 - Z2

def get_average_persistence_image(ph_list, xlim=None, ylim=None, 
                                  bw_method=None, 
                                  weighted=False, 
                                  resolution=100):
    """Plot the gaussian kernel of a population as an average of the ph diagrams that are given."""
    im_av = False
    k = 1
    if weighted:
        weights = [len(p) for p in ph_list]
        weights = np.array(weights, dtype=float) / np.max(weights)
    else:
        weights = [1 for _ in ph_list]

    for weight, ph in zip(weights, ph_list):
        if not isinstance(im_av, np.ndarray):
            try:
                im = persistence_image(ph, xlim=xlim, ylim=ylim, 
                                       bw_method=bw_method, 
                                       resolution=resolution)
                if not np.isnan(np.sum(im)):
                    im_av = weight * im
            except BaseException:  # pylint: disable=broad-except
                pass
        else:
            try:
                im = persistence_image(ph, xlim=xlim, ylim=ylim, 
                                       bw_method=bw_method, 
                                       resolution=resolution)                
                if not np.isnan(np.sum(im)):
                    im_av = np.add(im_av, weight * im)
                    k = k + 1
            except BaseException:  # pylint: disable=broad-except
                pass
    return im_av / k


def filter_pi_list(pis, tokeep = None, std_threshold = 1e-5):
    # pis is a matrix of flatten images i.e. a list of rows where each row is an image
    if tokeep is None:
        tokeep = np.where(
            np.std(pis, axis=0) >= std_threshold)[0]
        
    print(len(tokeep), 'pixels to keep over', pis.shape[1])
    mask = np.zeros(len(pis[0]), dtype=bool)
    mask[tokeep] = True
    pis_filtered = np.array([np.array(row[mask]) for row in pis])
    return pis_filtered, tokeep