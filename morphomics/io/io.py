"""
Python module that contains the functions
about reading and writing files.
"""
from __future__ import print_function

import os, glob
import numpy as np
import re
import pandas as pd

from operator import itemgetter
from morphomics.io.swc import SWC_DCT
from morphomics.cells.utils import LoadSWCError

# The following codes were adapted from TMD:
# https://github.com/BlueBrain/TMD


def read_swc(file_path, line_delimiter="\n"):
    """Load a swc file containing a list of sections, into a numpy.array format."""
    # Read all data from file.
    try:
        assert file_path.endswith((".swc"))
    except AssertionError:
        raise Warning("{} is not a valid swc file".format(file_path))
    except LoadSWCError:
        return np.nan
    
    with open(file_path, "r", encoding="utf-8") as f:
        read_data = f.read()

    # Split data per lines
    split_data = read_data.split(line_delimiter)
    # Clean data from comments and empty lines
    split_data = [a for a in split_data if "#" not in a]
    split_data = [a for a in split_data if a != ""]
    if len(split_data) < 2:
        return np.nan
    split_data = np.array(split_data)

    """Transform swc to np.array to be used in make_tree."""
    expected_data = re.compile(
        r"^\s*([-+]?\d*\.\d+|[-+]?\d+)"
        r"\s*([-+]?\d*\.\d+|[-+]?\d+)\s"
        r"*([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*$"
    )

    data = []
    for dpoint in split_data:
        if expected_data.match(dpoint.replace("\r", "")):
            segment_point = np.array(
                expected_data.match(dpoint.replace("\r", "")).groups(), dtype=float
            )
            # make the radius diameter
            segment_point[SWC_DCT["radius"]] = 2.0 * segment_point[SWC_DCT["radius"]]

            data.append(segment_point)
    swc_arr = np.array(data)
    return swc_arr


def load_ph(filename, delimiter=" "):
    """Load PH file in a `np.array`."""
    with open(filename, "r", encoding="utf-8") as f:
        ph = np.array([np.array(line.split(delimiter), dtype=float) for line in f])
    return ph


def get_info_frame(
    folder_location,
    extension=".swc",
    conditions=[],
):
    """Loads all data contained in input directory that ends in `extension`.

    Args:
        folder_location (string): the path to the main directory which contains .swc files
        extension (str, optional): last strings of the .swc files. NLMorphologyConverter results have "nl_corrected.swc" as extension. Defaults to ".swc".
        if .swc files are arranged in some pre-defined hierarchy:
        conditions (list of strings): list encapsulating the folder hierarchy in folder_location

    Returns:
        DataFrame: dataframe containing conditions, 'file_name', 'file_path' and 'neuron'
    """
    print(os.getcwd())

    print("You are now collecting the 3D reconstructions (.swc files) from this folder: \n%s\n"%folder_location)
    
    # get all the file paths in folder_location
    filepaths = glob.glob(
        "%s%s/*%s" % (folder_location, "/*" * len(conditions), extension)
    )
    print("Found %d files..." % len(filepaths))

    # convert the filepaths to array for metadata
    file_info = np.array(
        [_files.replace(folder_location, "").split("/")[1:] for _files in filepaths]
    )

    # create the dataframe for the population of cells
    info_frame = pd.DataFrame(data=file_info, columns=conditions + ["file_name"])
    info_frame["file_path"] = filepaths
    
    # print a sample of file names
    nb_files = len(filepaths)
    if nb_files > 0:
        print("Sample filenames:")
        for _ii in range(min(5, nb_files)): print(filepaths[_ii])
        print(" ")
    else:
        print("There are no files in folder_location! Check the folder_location in parameters file or the path to the parameters file.")

    return info_frame
