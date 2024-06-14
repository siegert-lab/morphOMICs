"""
Python module that contains the functions
about reading and writing files.
"""
from __future__ import print_function

import os, glob
import numpy as _np
import pandas as _pd
from scipy import sparse as sp
from scipy.sparse import csgraph as cs
from operator import itemgetter

# The following codes were adapted from TMD:
# https://github.com/BlueBrain/TMD
from tmd.io.swc import SWC_DCT
from tmd.io.swc import read_swc
from tmd.io.swc import swc_to_data
from tmd.io.io import make_tree
from tmd.io import io
from tmd.Population import Population
from tmd.Neuron import Neuron
from tmd.Tree import Tree
from tmd.Soma import Soma
from morphomics.utils import tree_type as td
from morphomics.utils import save_obj
from tmd.Topology import analysis
from tmd.Topology import methods


# Definition of tree types
TYPE_DCT = {"soma": 1, "basal": 3, "apical": 4, "axon": 2, "glia": 7}


class LoadNeuronError(Exception):
    """
    Captures the exception of failing to load a single neuron
    """

def load_neuron(
    input_file,
    line_delimiter="\n",
    soma_type=None,
    tree_types=None,
    remove_duplicates=True,
):
    """Io method to load an swc into a Neuron object.

    TODO: Check if tree is connected to soma, otherwise do
    not include it in neuron structure and warn the user
    that there are disconnected components
    """

    tree_types_final = td.copy()
    if tree_types is not None:
        tree_types_final.update(tree_types)

    # Definition of swc types from type_dict function
    if soma_type is None:
        soma_index = TYPE_DCT["soma"]
    else:
        soma_index = soma_type

    # Make neuron with correct filename and load data
    if os.path.splitext(input_file)[-1] == ".swc":
        data = swc_to_data(
            read_swc(input_file=input_file, line_delimiter=line_delimiter)
        )
        neuron = Neuron.Neuron(name=input_file.replace(".swc", ""))
    try:
        soma_ids = _np.where(_np.transpose(data)[1] == soma_index)[0]
    except IndexError:
        raise LoadNeuronError("Soma points not in the expected format")
    # print(os.path.splitext(input_file)[-2:], len(soma_ids))

    # Extract soma information from swc
    soma = Soma.Soma(
        x=_np.transpose(data)[SWC_DCT["x"]][soma_ids],
        y=_np.transpose(data)[SWC_DCT["y"]][soma_ids],
        z=_np.transpose(data)[SWC_DCT["z"]][soma_ids],
        d=_np.transpose(data)[SWC_DCT["radius"]][soma_ids],
    )

    # Save soma in Neuron
    neuron.set_soma(soma)
    p = _np.array(_np.transpose(data)[6], dtype=int) - _np.transpose(data)[0][0]
    # return p, soma_ids
    try:
        dA = sp.csr_matrix(
            (
                _np.ones(len(p) - len(soma_ids)),
                (range(len(soma_ids), len(p)), p[len(soma_ids) :]),
            ),
            shape=(len(p), len(p)),
        )
    except Exception:
        raise LoadNeuronError(
            "Cannot create connectivity, nodes not connected correctly."
        )

    # assuming soma points are in the beginning of the file.
    comp = cs.connected_components(dA[len(soma_ids) :, len(soma_ids) :])

    # Extract trees
    for i in range(comp[0]):
        tree_ids = _np.where(comp[1] == i)[0] + len(soma_ids)
        tree = make_tree(data[tree_ids])
        neuron.append_tree(tree, tree_types=tree_types_final)

    return neuron

def load_population(neurons, tree_types=None, name=None):
    """Loads all data of recognised format (.swc) into a Population object.

    Args:
        neurons (list, tuple): directory or a list of .swc files to load
        tree_types (int, optional): see TYPE_DCT for dictionary of tree types. Defaults to None.
        name (str, optional): custom name for the population. Defaults to None.

    Returns:
        pop (Population object): morphology
        files (str, list): filenames of the pop object
    """
    if isinstance(neurons, (list, tuple)):
        files = neurons
        name = name if name is not None else "Population"
    elif os.path.isdir(neurons):  # Assumes given input is a directory
        files = [os.path.join(neurons, neuron_dir) for neuron_dir in os.listdir(neurons)]
        name = name if name is not None else os.path.basename(neurons)
    elif os.path.isfile(neurons):  # Assumes given input is a file
        files = [neurons]
        name = name if name is not None else os.path.basename(neurons)
    else:
        raise TypeError(
            "The format of the given neurons is not supported. "
            "Expected an iterable of files, or a directory, or a single morphology file. "
            f"Got: {neurons}"
        )
    pop = Population.Population(name=name)
    ### specific to morphomics
    # update_stamps = filename[]
    fnames = []
    failed_fnames = []
    
    L = len(files)
    stamps = [int(0.25*L), int(0.5*L), int(0.75*L)]
    file_stamps = itemgetter(*stamps)(files)
    _i_stamp = 1
    ###
    for filename in files:
        try:
            assert filename.endswith((".swc"))
            neuron = load_neuron(input_file = filename, tree_types=tree_types)
            pop.append_neuron(neuron)
            fnames.append(filename)
        except AssertionError:
            raise Warning("{} is not a valid swc file".format(filename))
        except LoadNeuronError:
            failed_fnames.append(filename)
        if filename in file_stamps:
            print("You have loaded %d%% chunk of the data..."%((_i_stamp/4)*100))
            _i_stamp += 1

    return pop, fnames, failed_fnames


def exclude_single_branch_ph(neuron, feature="radial_distances"):
    """
    Calculates persistence diagram and only considers
    those with more than one bar
    """
    phs = []
    for tree in neuron.neurites:
        p = methods.get_persistence_diagram(tree, feature="radial_distances")
        if len(p) > 1:
            phs.append(p)
    return phs


def get_barcodes_from_df(
    info_frame, filtration_function="radial_distances", save_filename=None
):
    """Converts .swc files in info_frame into persistence barcodes

    Args:
        info_frame (DataFrame): contains at least the filenames (as a column) of all .swc files to be converted
        filtration_function (str, optional): filter function for TMD. Can either be "radial_distances" or "path_distances". Defaults to "radial_distances".
        save_filename (str, optional): filename where to save the DataFrame with the morphologies and barcodes. Defaults to None.

    Returns:
        DataFrame: dataframe containing 'Morphologies' and 'Barcodes' for every .swc file in 'Filenames'
    """
    assert (
        "path_to_file" in info_frame.keys()
    ), "`path_to_file` must be a column in the info_frame DataFrame"

    info_frame["tree"] = None
    info_frame["barcodes"] = None

    _fnames = list(info_frame["path_to_file"].values)
    pops, files, failed_files = load_population(_fnames)
    print("...finished loading morphologies and barcodes...")
    print(f"! Warning: You have {len(failed_files)} .swc files that did not load. Potentially, empty files. Please check *-FailedFiles")
    
    for ii in _np.arange(len(files)):
        _idx = info_frame.loc[info_frame["path_to_file"] == files[ii]].index[0]
        info_frame = info_frame.copy()
        info_frame.loc[_idx, "tree"] = pops.neurons[ii]
        info_frame.loc[_idx, "barcodes"] = analysis.collapse(
            exclude_single_branch_ph(pops.neurons[ii], feature=filtration_function)
        )
    if save_filename is not None:
        save_obj(info_frame, save_filename)
        _np.savetxt("%s-FailedFiles.txt" % (save_filename), failed_files, delimiter='\n', fmt="%s")

    return info_frame


def load_data(
    folder_location,
    extension=".swc",
    filtration_function="radial_distances",
    save_filename=None,
    conditions=[],
    separated_by=None,
):
    """Loads all data contained in input directory that ends in `extension`.

    Args:
        folder_location (string): the path to the main directory which contains .swc files
        extension (str, optional): last strings of the .swc files. NLMorphologyConverter results have "nl_corrected.swc" as extension. Defaults to ".swc".
        filtration_function (str, optional): filter function for TMD. Can either be "radial_distances" or "path_distances". Defaults to "radial_distances".
        save_filename (_type_, optional): filename where to save the DataFrame with the morphologies and barcodes. Defaults to None.

        if .swc files are arranged in some pre-defined hierarchy:
        conditions (list of strings): list encapsulating the folder hierarchy in folder_location
        separated_by (_type_, optional): an element in conditions which will be used to break down the returned DataFrame. Defaults to None.

    Returns:
        DataFrame: dataframe containing conditions, 'Filenames', 'Morphologies' and 'Barcodes'
        for every .swc file in the `folder_location` separated according to `separated_by` (if given)
    """

    print("You are now loading the 3D reconstructions (.swc files) from this folder: \n%s\n"%folder_location)
    
    assert filtration_function in [
        "radial_distances",
        "path_distances",
    ], "Currently, TMD is only implemented with either radial_distances or path_distances"

    # get all the file names in folder_location
    filenames = glob.glob(
        "%s%s/*%s" % (folder_location, "/*" * len(conditions), extension)
    )
    # print a sample of file names
    nb_files = len(filenames)
    if nb_files > 0:
        print("Sample filenames:")
        for _ii in range(min(5, nb_files)): print(filenames[_ii])
        print(" ")
    else:
        print("There are no files in folder_location! Check the folder_location in parameters file or the path to the parameters file.")
    
    # convert the filenames to array for metadata
    file_info = _np.array(
        [_files.replace(folder_location, "").split("/")[1:] for _files in filenames]
    )

    _info_frame = _pd.DataFrame(data=file_info, columns=conditions + ["_file_name"])
    _info_frame["path_to_file"] = filenames
    print("Found %d files..." % len(filenames))
    
    if separated_by is not None:
        assert (
            len(conditions) > 1
        ), "`conditions` must have more than one element. Otherwise, remove `separated_by` argument"
        assert separated_by in conditions, "`separated_by` must be in `conditions`"

        cond_values = _info_frame[separated_by].unique()
        info_frame = {}

        print("Separating DataFrame into %s..." % separated_by)
        print("There are %d values for the separating condition..." % len(cond_values))

        for _v in cond_values:
            print("...processing %s" % _v)
            _sub_info_frame = (
                _info_frame.loc[_info_frame[separated_by] == _v]
                .copy()
                .reset_index(drop=True)
            )

            if save_filename is not None:
                _save_filename = "%s.%s-%s" % (save_filename, separated_by, _v)
            info_frame[_v] = get_barcodes_from_df(
                _sub_info_frame, filtration_function=filtration_function, save_filename=_save_filename
            )

        info_frame = _pd.concat([info_frame[_c] for _c in cond_values], ignore_index=True)
            
    else:
        info_frame = get_barcodes_from_df(
            _info_frame, filtration_function=filtration_function, save_filename=save_filename
        )
        
    if save_filename is not None:
        save_obj(info_frame, save_filename)

    return info_frame