import numpy as np
import os
import glob

# Library for Morphometrics
import pylmeasure
# Library for Sholl Analysis
from morphon import Morph, select, sholl
from math import isclose

# You can find the doc of the L-measures here: http://cng.gmu.edu:8080/Lm/help/index.htm
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
    # "XYZ", # This one is not in the list, idk why it was here.
    #"Branch_Order", # This one does not work for some .swc. 
                    # I think it doesn't work when the tree has one branch or is two small.
    "Terminal_degree",
    "TerminalSegment",
    "Taper_1",
    "Taper_2",
    "Branch_pathlength",
    "Contraction",
    "Fragmentation",
    #"Daughter_Ratio",                  # All these ones don't work for some .swc
    # "Parent_Daughter_Ratio",
    # "Partition_asymmetry",
    "Rall_Power",
#     "Pk",
#     "Pk_classic",
#     "Pk_2",
#     "Bif_ampl_local",
#     "Bif_ampl_remote",
#     "Bif_tilt_local",
#     "Bif_tilt_remote",
#     "Bif_torque_local",
#     "Bif_torque_remote",
#     "Last_parent_diam",
#     "Diam_threshold",
#     "HillmanThreshold",
#     "Helix",
#     "Fractal_Dim",
]

def remove_not_lmeasure(Lm_functions, Lm_quantities=None):
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
        if Lm_quantities is not None:
            Lm_quantities = Lm_quantities[np.whree(test_function_availability)[0]]
    else:
        print(
            "All of the functions in Morphometrics.Lmeasure_functions are available in Lmeasure!"
        )
    return Lm_functions, Lm_quantities


def get_quantities(Lmeasure_functions, filename_list):
    distribution = ['Minimum', 'Average', 'Maximum', 'StdDev']
    nb_quantiles = len(distribution)
    scalar = ['TotalSum']
    Lm_functions = []
    Lm_quantities = []
    test_output = pylmeasure.getMeasure(Lmeasure_functions, filename_list)
    for i, f in enumerate(Lmeasure_functions):
        # If there is no variance it means that the morphometrics is not a distribution
        if isclose(test_output[i]['WholeCellMeasuresDict'][0]['StdDev'], 0, abs_tol=1e-5):
            Lm_quantities.extend(scalar)
            Lm_functions.append(f)
        else:
            Lm_quantities.extend(distribution)
            repreated_f = nb_quantiles * [f]
            Lm_functions.extend(repreated_f)
    return Lm_functions, Lm_quantities


def create_Lm_functions(Lmeasure_functions = None, filename_list = None):
    '''The function `create_Lm_functions` checks if the listed functions are valid Lmeasure functions and
    removes any that are not available.

    
    Returns
    -------
        The function `create_Lm_functions` returns two arrays: `Lm_functions` and `Lm_quantities`. These
    arrays contain the valid Lmeasure functions and their corresponding quantities (ex: Average, Min ...) after checking if the
    functions listed in `Lmeasure_functions` are available in `Lmeasure`. If any functions are not
    available, they are removed from the arrays before returning them.
    
    '''
    # If Lmeasure_functions not given, computes all the morphometrics
    if Lmeasure_functions is None:
        Lmeasure_functions = Lmeasure_FunctionList
    # If no quantities are given,
    # all the appropriate (quantiles for distributions or number for unique scalar) quantities are computed
    if type(Lmeasure_functions[0]) is not list:
        Lm_functions, _ = remove_not_lmeasure(Lmeasure_functions)
        Lm_functions, Lm_quantities = get_quantities(Lmeasure_functions, filename_list)
    else: 
        Lm_functions = np.array(Lmeasure_functions)[:, 0]
        Lm_quantities = np.array(Lmeasure_functions)[:, 1]
        Lm_functions, Lm_quantities = remove_not_lmeasure(Lm_functions, Lm_quantities)

    return Lm_functions, Lm_quantities


def compute_morphometrics(filenames_to_process, Lm_functions, Lm_quantities):
    nb_files = len(filenames_to_process)
    print("Running pyLmeasure with %d files..." % nb_files)
    print(Lm_functions)
    Lm_output = pylmeasure.getMeasure(Lm_functions, filenames_to_process)
    
    print("Summarizing morphometric quantities into an array...")
    # Collecting all results
    morphometric_quantities = []
    for file_ind in np.arange(nb_files):
        _morphometric_quantities = []
        for morpho_ind, quantity in enumerate(Lm_quantities):
            _morphometric_quantities.append(
                Lm_output[morpho_ind]["WholeCellMeasuresDict"][file_ind][quantity]
            )
        morphometric_quantities.append(_morphometric_quantities)

    return np.array(morphometric_quantities) 


def compute_morphometrics_distribution(filenames_to_process, Lm_functions, bins=20):
    nb_files = len(filenames_to_process)
    print("Running pyLmeasure with %d files..." % nb_files)
    Lm_output = pylmeasure.getMeasureDistribution(Lm_functions, filenames_to_process, nBins=bins)

    print("Summarizing morphometric quantities into an array...")
    # Collecting all results
    morphometric_quantities = []
    morphometric_bins = []
    for file_ind in np.arange(nb_files):
        _morphometric_quantities = []
        _morphometric_bins = []
        for morpho_ind, _ in enumerate(Lm_functions):
            _morphometric_quantities.extend(
                Lm_output[morpho_ind]["measure1BinCounts"][file_ind]
            )
            _morphometric_bins.extend(
                Lm_output[morpho_ind]["measure1BinCentres"][file_ind]
            )
        morphometric_quantities.append(_morphometric_quantities)
        morphometric_bins.append(_morphometric_bins)

    return np.array(morphometric_quantities), np.array(morphometric_bins)


def compute_lmeasures(filenames, 
                      Lmeasure_functions = None, 
                      histogram = False, 
                      bins = 100, 
                      tmp_folder = ''):
    '''This Python function processes SWC files for morphometric analysis using pyLmeasure and returns the
    calculated morphometric quantities.
    '''

    if "nt" in os.name:
        char0 = '%s\\tmp%d.swc'
        char1 = "\\"
    else:
        char0 = '%s/tmp%d.swc'
        char1 = "/"

    # Lmeasure does not like filenames with spaces
    # So they are renamed temporarly 
    filenames_to_process = []
    tmp_ind = 0
    for filename in filenames:
        space_in_filename = " " in filename
        if space_in_filename:
            os.system("cp '%s' '%s'" % (filename, char0 % (tmp_folder, tmp_ind)))
            filename = char0 % (tmp_folder, tmp_ind)
            tmp_ind = tmp_ind + 1
        filenames_to_process.append(filename)
    if tmp_ind > 0:
        print("There were %d files in the tmp_directory" % (tmp_ind-1))

    print('Define Lm functions')
    filename = [filenames_to_process[0]]
    Lm_functions, Lm_quantities = create_Lm_functions(Lmeasure_functions, filename)
    print('Define Lm functions done')

    # Get morphometrics    
    if not histogram:
        morphometric_quantities = compute_morphometrics(filenames_to_process, Lm_functions, Lm_quantities)
    else:
        # Keeps functions that output distribution
        Lm_functions = [
            f for f, q in zip(Lm_functions, Lm_quantities) if q == "StdDev"
        ]
        morphometric_quantities, morphometric_bins = compute_morphometrics_distribution(filenames_to_process, Lm_functions, bins=bins)
    
    # Remove temporary files
    if tmp_ind > 0:
        print("Removing temporary files...")
        for file_path in glob.glob(f"{tmp_folder}{char1}tmp*.swc"):
            os.remove(file_path)
    if not histogram:
        return morphometric_quantities, Lm_functions, Lm_quantities
    else:
        return morphometric_quantities, Lm_functions, morphometric_bins
    

def compute_sholl_curves(_files, _Sholl_radius, _type=None):
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