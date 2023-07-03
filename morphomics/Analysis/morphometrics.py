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