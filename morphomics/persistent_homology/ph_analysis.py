import numpy as np

def get_limits(phs_list):
    """Returns the x-y coordinates limits (min, max) for a list of persistence diagrams."""
    # if any((isinstance(ph[0], list) for ph in phs_list)):
    #     phs = [list(ph_bar) for ph in phs_list for ph_bar in ph]
    # else:
    #     phs = phs_list
    phs = np.vstack(phs_list)
    xlim = [min(np.transpose(phs)[0]), max(np.transpose(phs)[0])]
    ylim = [min(np.transpose(phs)[1]), max(np.transpose(phs)[1])]
    return xlim, ylim


def get_bifurcations(ph):
    """Return the bifurcations from the diagram."""
    return np.array(ph)[:, 1]


def get_terminations(ph):
    """Return the terminations from the diagram."""
    return np.array(ph)[:, 0]


def get_lengths(ph, abs = True):
    """Return the length of the bars from the diagram."""
    lengths = ph[:,1]-ph[:,0] 
    if abs:
        lengths = np.abs(lengths)
    return lengths


def get_total_length(ph, abs = True):
    """Calculate the total length of a barcode.

    The total length is computed by summing the length of each bar.
    This should be equivalent to the total length of the tree if the barcode represents path
    distances.
    """
    return np.sum(get_lengths(ph, abs))

def closest_ph(ph_list, target_extent, method="from_above"):
    """Get index of the persistent homology in the ph_list closest to a target extent.

    For each ph the maximum extent is computed and compared to the target_extent according to the
    selected method:

    * `from_above`: smallest maximum extent that is greater or equal than target_extent.
    * `from_below`: biggest maximum extent that is smaller or equal than target_extent.
    * `nearest`: closest by absolute value.
    """
    n_bars = len(ph_list)
    max_extents = np.asarray([max(get_lengths(ph)) for ph in ph_list])

    sorted_indices = np.argsort(max_extents, kind="mergesort")
    sorted_extents = max_extents[sorted_indices]

    if method == "from_above":
        above = np.searchsorted(sorted_extents, target_extent, side="right")

        # if target extent is close to current one, return this instead
        if above >= 1 and np.isclose(sorted_extents[above - 1], target_extent):
            closest_index = above - 1
        else:
            closest_index = above

        closest_index = np.clip(closest_index, 0, n_bars - 1)

    elif method == "from_below":
        below = np.searchsorted(sorted_extents, target_extent, side="left")

        # if target extent is close to current one, return this instead
        if below < n_bars and np.isclose(sorted_extents[below], target_extent):
            closest_index = below
        else:
            closest_index = below - 1

        closest_index = np.clip(closest_index, 0, n_bars - 1)

    elif method == "nearest":
        below = np.searchsorted(sorted_extents, target_extent, side="left")
        pos = np.clip(below, 0, n_bars - 2)

        closest_index = min((pos, pos + 1), key=lambda i: abs(sorted_extents[i] - target_extent))

    else:
        raise TypeError(f"Unknown method {method} for closest_ph")

    return sorted_indices[closest_index]