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
from morphomics.io.swc import SWC_DCT
from morphomics.io.swc import read_swc
from morphomics.io.swc import swc_to_data
from morphomics.Neuron import Neuron
from morphomics.Tree import Tree
from morphomics.Soma import Soma
from morphomics.Population import Population
from morphomics.utils import tree_type as td
from morphomics.utils import save_obj
from morphomics.Topology import analysis
from morphomics.Topology import methods


# Definition of tree types
TYPE_DCT = {"soma": 1, "basal": 3, "apical": 4, "axon": 2, "glia": 7}


class LoadNeuronError(Exception):
    """
    Captures the exception of failing to load a single neuron
    """


def make_tree(data):
    """
    Make tree structure from loaded data.
    Returns a tree of morphomics.Tree type.
    """
    tr_data = _np.transpose(data)

    parents = [
        _np.where(tr_data[0] == i)[0][0]
        if len(_np.where(tr_data[0] == i)[0]) > 0
        else -1
        for i in tr_data[6]
    ]

    return Tree.Tree(
        x=tr_data[SWC_DCT["x"]],
        y=tr_data[SWC_DCT["y"]],
        z=tr_data[SWC_DCT["z"]],
        d=tr_data[SWC_DCT["radius"]],
        t=tr_data[SWC_DCT["type"]],
        p=parents,
    )


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
        neuron.append_tree(tree, td=tree_types_final)

    return neuron


def load_ph_file(filename, delimiter=" "):
    """Load PH file in a np.array

    Args:
        filename (str): Path to the PH file
        delimiter (str): separator to use. Defaults to " ".

    Returns:
        numpy array: persistence barcode
    """
    f = open(filename, "r")
    ph = _np.array([_np.array(line.split(delimiter), dtype=float) for line in f])
    f.close()
    return ph


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
        files = [os.path.join(neurons, l) for l in os.listdir(neurons)]
        name = name if name is not None else os.path.basename(neurons)

    pop = Population.Population(name=name)

    # update_stamps = filename[]
    fnames = []
    failed_fnames = []
    
    L = len(files)
    stamps = [int(0.25*L), int(0.5*L), int(0.75*L)]
    file_stamps = itemgetter(*stamps)(files)
    _i_stamp = 1
    for filename in files:
        try:
            assert filename.endswith((".swc"))
            pop.append_neuron(load_neuron(filename, tree_types=tree_types))
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
    only those with more than one bar
    """
    phs = []
    for tree in neuron.neurites:
        p = methods.get_persistence_diagram(tree, feature="radial_distances")
        if len(p) > 1:
            phs.append(p)
    return phs


def get_barcodes_from_df(
    info_frame, barcode_filter="radial_distances", save_filename=None
):
    """Converts .swc files in info_frame into persistence barcodes

    Args:
        info_frame (DataFrame): contains at least the filenames (as a column) of all .swc files to be converted
        barcode_filter (str, optional): filter function for TMD. Can either be "radial_distances" or "path_distances". Defaults to "radial_distances".
        save_filename (str, optional): filename where to save the DataFrame with the morphologies and barcodes. Defaults to None.

    Returns:
        DataFrame: dataframe containing 'Morphologies' and 'Barcodes' for every .swc file in 'Filenames'
    """
    assert (
        "Filenames" in info_frame.keys()
    ), "`Filenames` must be a column in the info_frame DataFrame"

    info_frame["Morphologies"] = None
    info_frame["Barcodes"] = None

    _fnames = list(info_frame["Filenames"].values)
    pops, files, failed_files = load_population(_fnames)
    print("...finished loading morphologies and barcodes...")
    print(f"! Warning: You have {len(failed_files)} .swc files that did not load. Potentially, empty files. Please check *-FailedFiles")
    
    for ii in _np.arange(len(files)):
        _idx = info_frame.loc[info_frame["Filenames"] == files[ii]].index[0]
        info_frame = info_frame.copy()
        info_frame.loc[_idx, "Morphologies"] = pops.neurons[ii]
        info_frame.loc[_idx, "Barcodes"] = analysis.collapse(
            exclude_single_branch_ph(pops.neurons[ii], feature=barcode_filter)
        )

    if save_filename is not None:
        save_obj(info_frame, save_filename)
        _np.savetxt("%s-FailedFiles.txt" % (save_filename), failed_files, delimiter='\n', fmt="%s")

    return info_frame


def load_data(
    folder_location,
    extension=".swc",
    barcode_filter="radial_distances",
    save_filename=None,
    conditions=[],
    separated_by=None,
):
    """Loads all data contained in input directory that ends in `extension`.

    Args:
        folder_location (string): the path to the main directory which contains .swc files
        extension (str, optional): last strings of the .swc files. NLMorphologyConverter results have "nl_corrected.swc" as extension. Defaults to ".swc".
        barcode_filter (str, optional): filter function for TMD. Can either be "radial_distances" or "path_distances". Defaults to "radial_distances".
        save_filename (_type_, optional): filename where to save the DataFrame with the morphologies and barcodes. Defaults to None.

        if .swc files are arranged in some pre-defined hierarchy:
        conditions (list of strings): list encapsulating the folder hierarchy in folder_location
        separated_by (_type_, optional): an element in conditions which will be used to break down the returned DataFrame. Defaults to None.

    Returns:
        DataFrame: dataframe containing conditions, 'Filenames', 'Morphologies' and 'Barcodes'
        for every .swc file in the `folder_location` separated according to `separated_by` (if given)
    """

    print("You are now loading the 3D reconstructions (.swc files) from this folder: \n%s\n"%folder_location)
    
    assert barcode_filter in [
        "radial_distances",
        "path_distances",
    ], "Currently, TMD is only implemented with either radial_distances or path_distances"

    # getting all the files in folder_location
    filenames = glob.glob(
        "%s%s/*%s" % (folder_location, "/*" * len(conditions), extension)
    )
    if len(filenames)> 0:
        print("Sample filenames:")
        for _ii in range(min(5,len(filenames))): print(filenames[_ii])
        print(" ")
    else:
        print("There are no files in folder_location! Check the folder_location in parameters file or the path to the parameters file.")
    
    # convert the filenames to array for metadata
    file_info = _np.array(
        [_files.replace(folder_location, "").split("/")[1:] for _files in filenames]
    )
    _info_frame = _pd.DataFrame(data=file_info, columns=conditions + ["_files"])
    _info_frame["Filenames"] = filenames
    print("Found %d files..." % len(filenames))

    if separated_by is not None:
        assert (
            len(conditions) > 1
        ), "`conditions` must have more than one element. Otherwise, remove `separated_by` argument"
        assert separated_by in conditions, "`separated_by` must be in `conditions`"

        conds = _info_frame[separated_by].unique()
        info_frame = {}

        print("Separating DataFrame into %s..." % separated_by)
        print("There are %d conditions..." % len(conds))

        for _c in conds:
            print("...processing %s" % _c)
            _InfoFrame = (
                _info_frame.loc[_info_frame[separated_by] == _c]
                .copy()
                .reset_index(drop=True)
            )

            if save_filename is not None:
                _save_filename = "%s.%s-%s" % (save_filename, separated_by, _c)
            info_frame[_c] = get_barcodes_from_df(
                _InfoFrame, barcode_filter=barcode_filter, save_filename=_save_filename
            )

        info_frame = _pd.concat([info_frame[_c] for _c in conds], ignore_index=True)
            
    else:
        info_frame = get_barcodes_from_df(
            _info_frame, barcode_filter=barcode_filter, save_filename=save_filename
        )
        
    if save_filename is not None:
        save_obj(info_frame, save_filename)

    return info_frame
