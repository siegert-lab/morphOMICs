import glob
import numpy as _np
import pandas as _pd
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

    # get all the files in folder_location
    filenames = glob.glob(
        "%s%s/*%s" % (folder_location, "/*" * len(conditions), extension)
    )
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
                _sub_info_frame, barcode_filter=barcode_filter, save_filename=_save_filename
            )

        info_frame = _pd.concat([info_frame[_c] for _c in cond_values], ignore_index=True)
            
    else:
        info_frame = get_barcodes_from_df(
            _info_frame, barcode_filter=barcode_filter, save_filename=save_filename
        )
        
    if save_filename is not None:
        save_obj(info_frame, save_filename)

    return info_frame