"""
morphomics : classical mophological analyses, morphometrics and Sholl analysis

Author: Ryan Cubero
"""
import numpy as np
import os
from morphon import Morph, select, sholl
import pylmeasure

Lmeasure_FunctionList = [
    "Soma_Surface",
    "N_stems",
    "N_bifs",
    "N_branch",
    "N_tips",
    "Width",
    "Height",
    "Depth",
    "Type",
    "Diameter",
    "Diameter_pow",
    "Length",
    "Surface",
    "SectionArea",
    "Volume",
    "EucDistance",
    "PathDistance",
    "XYZ",
    "Branch_Order",
    "Terminal_degree",
    "TerminalSegment",
    "Taper_1",
    "Taper_2",
    "Branch_pathlength",
    "Contraction",
    "Fragmentation",
    "Daughter_Ratio",
    "Parent_Daughter_Ratio",
    "Partition_asymmetry",
    "Rall_Power",
    "Pk",
    "Pk_classic",
    "Pk_2",
    "Bif_ampl_local",
    "Bif_ampl_remote",
    "Bif_tilt_local",
    "Bif_tilt_remote",
    "Bif_torque_local",
    "Bif_torque_remote",
    "Last_parent_diam",
    "Diam_threshold",
    "HillmanThreshold",
    "Helix",
    "Fractal_Dim",
]


def calculate_sholl_curves(_files, _Sholl_radius, _type=None):
    '''The function `calculate_sholl_curves` loads morphological data, selects specific sections based on
    type, and calculates Sholl curves with a specified radius step.
    
    Parameters
    ----------
    _files
        The `_files` parameter in the `calculate_sholl_curves` function is used to specify the file or
    files that contain the morphological data to be analyzed. This function loads the morphological data
    from the specified file(s) using the `m.load(_files)` method.
    _Sholl_radius
        The `_Sholl_radius` parameter in the `calculate_sholl_curves` function represents the radius
    increment used for Sholl analysis. It determines the spacing between the concentric circles around
    the neuron's soma within which intersections with neuronal processes are counted. This parameter
    controls the resolution of the Sholl analysis
    _type
        The `_type` parameter in the `calculate_sholl_curves` function is used to specify the types of
    sections to include in the Sholl analysis. It is a list of integers representing the types of
    sections. If `_type` is not provided or is set to 0, it will default
    
    Returns
    -------
        The function `calculate_sholl_curves` returns a NumPy array containing the Sholl radii and
    corresponding Sholl values calculated based on the input parameters `_files`, `_Sholl_radius`, and
    `_type`.
    
    '''
    m = Morph()
    m.load(_files)

    if (_type is None) or (_type == 0):
        _type = ",".join(np.arange(2,9).astype('str'))
    types = set(int(item) for item in _type.replace(',', ' ').split(' '))
    idents = list()
    for sec in select(m, m.sections(), types=types, orders=[], degrees=[]):
        idents.extend(sec)
    radx, crox = sholl(m, idents, step=_Sholl_radius)
    return np.array(list((x,y) for (x,y) in zip(radx,crox)))


def create_Lm_functions(Lmeasure_functions):
    '''The function `create_Lm_functions` checks if the listed functions are valid Lmeasure functions and
    removes any that are not available.
    
    Parameters
    ----------
    Lmeasure_functions
        It seems like you haven't provided the value for the `Lmeasure_functions` parameter. In order to
    assist you further with the `create_Lm_functions` function, could you please provide the list of
    Lmeasure functions that you want to work with?
    
    Returns
    -------
        The function `create_Lm_functions` returns two arrays: `Lm_functions` and `Lm_quantities`. These
    arrays contain the valid Lmeasure functions and their corresponding quantities after checking if the
    functions listed in `Lmeasure_functions` are available in `Lmeasure`. If any functions are not
    available, they are removed from the arrays before returning them.
    
    '''
    Lm_functions = np.array(Lmeasure_functions)[:, 0]
    Lm_quantities = np.array(Lmeasure_functions)[:, 1]

    # Check if the functions listed in Morphometrics.Lmeasure_functions are valid Lmeasure functions
    test_function_availability = np.array(
        [func_i in Lmeasure_FunctionList for func_i in Lm_functions]
    )

    if len(np.where(~test_function_availability)[0]) > 0:
        print(
            "The following functions are not available in Lmeasure: %s"
            % (", ".join(Lm_functions[np.where(~test_function_availability)[0]]))
        )
        print("Please check Morphometrics.Lmeasure_functions in the parameter file")
        print("Removing the non-available function")
        Lm_functions = Lm_functions[np.where(test_function_availability)[0]]
        Lm_quantities = Lm_quantities[np.where(test_function_availability)[0]]
    else:
        print(
            "All of the functions in Morphometrics.Lmeasure_functions are available in Lmeasure!"
        )

    return Lm_functions, Lm_quantities


def calculate_morphometrics(filenames, tmp_folder, Lm_functions, Lm_quantities):
    '''This Python function processes SWC files for morphometric analysis using pyLmeasure and returns the
    calculated morphometric quantities.
    
    Parameters
    ----------
    filenames
        Please provide the list of filenames that you want to process using the `calculate_morphometrics`
    function.
    tmp_folder
        The `tmp_folder` parameter in the `calculate_morphometrics` function is a string that represents
    the path to a temporary folder where SWC files with spaces in their filenames will be copied for
    processing. This temporary folder is used to handle filenames with spaces, as Lmeasure does not
    support filenames with
    Lm_functions
        Lm_functions is a list of L-measure functions that you want to calculate for the morphometrics.
    These functions are specific measurements or analyses that you want to perform on the SWC files.
    Examples of Lm_functions could include 'TotalSurfaceArea', 'TotalVolume', 'NumBifur
    Lm_quantities
        Lm_quantities is a list of morphometric quantities that you want to extract from the SWC files
    using the Lmeasure functions. These quantities could include measurements such as total length,
    surface area, volume, etc.
    
    Returns
    -------
        The function `calculate_morphometrics` returns two values: 
    1. `files_to_process`: a list of filenames that have been processed
    2. `morphometric_quantities`: an array containing the morphometric quantities calculated for each
    file in `files_to_process`
    
    '''
    # Lmeasure does not like filenames with spaces
    files_to_process = []
    tmp_ind = 0
   
    for filename in filenames:
        space_in_filename = " " in filename
        if space_in_filename:
            os.system("cp '%s' '%s/tmp%d.swc'" % (filename, tmp_folder, tmp_ind))
            filename = "%s/tmp%d.swc" % (tmp_folder, tmp_ind)
            tmp_ind = tmp_ind + 1
        files_to_process.append(filename)
    if tmp_ind > 0:
        print("There were %d files in the tmp_directory" % (tmp_ind-1))

    print("Running pyLmeasure with %d files..." % len(files_to_process))
    # implementation of pylmeasure
    LMOutput = pylmeasure.getMeasure(Lm_functions, files_to_process)
    
    print("Summarizing morphometric quantities into an array...")
    # collecting all results
    morphometric_quantities = []
    for file_ind in np.arange(len(files_to_process)):
        _morphometric_quantities = []
        for morpho_ind, quantity in enumerate(Lm_quantities):
            _morphometric_quantities.append(
                LMOutput[morpho_ind]["WholeCellMeasuresDict"][file_ind][quantity]
            )
        morphometric_quantities.append(_morphometric_quantities)
        
    if tmp_ind > 0:
        print("Removing temporary files...")
        os.system("rm '%s/tmp*.swc'" % (filename, tmp_folder))
    
    return files_to_process, np.array(morphometric_quantities)