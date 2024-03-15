"""
morphomics : plotting tools

Author: Ryan Cubero
"""
import numpy as np
import pickle as pkl

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, convex_hull_plot_2d

import ipyvolume as ipv


   
def plot_convex_hulled(
    X_reduced,
    foreground_regions,
    background_regions,
    brain_conds,
    brain_labels,
    conditions,
    sizes,
    colors,
    pre_cond,
    savefile,
): 
    '''The function `plot_convex_hulled_MF_dev` generates plots with convex hulls for different conditions
    and genders based on input data.
    
    Parameters
    ----------
    X_reduced
        X_reduced is a 2D numpy array containing the reduced data points in two dimensions after
    dimensionality reduction (e.g., UMAP or PCA).
    foreground_regions
        The `foreground_regions` parameter in the `plot_convex_hulled` function represents the foreground
    regions in the plot where data points will be highlighted or emphasized. These regions are typically
    the main areas of interest that you want to visually distinguish from the background regions.
    background_regions
        The `background_regions` parameter in the `plot_convex_hulled` function refers to the regions in
    the background where the convex hull will be plotted. These regions are used to identify specific
    areas in the dataset for visualization and analysis.
    brain_conds
        The `brain_conds` parameter represents the conditions associated with brain data. It is used in the
    provided functions `plot_convex_hulled` and `plot_convex_hulled_MF_dev` to filter data based on
    specific conditions.
    brain_labels
        The `brain_labels` parameter likely represents the labels assigned to different regions in the
    brain. These labels could indicate specific anatomical regions, functional regions, or any other
    categorization used in the context of the data being analyzed.
    conditions
        The `conditions` parameter in the provided functions represents a list of conditions or categories
    that are being analyzed or plotted in the visualization. These conditions could be different
    experimental groups, treatment conditions, or any other categorical variable that is relevant to the
    data being visualized. The functions use this parameter to iterate over
    sizes
        The `sizes` parameter in the `plot_convex_hulled` and `plot_convex_hulled_MF_dev` functions is a
    dictionary that maps conditions to the size of markers to be used for each condition in the scatter
    plot. The size of markers can be adjusted based on the condition being
    colors
        The `colors` parameter in both functions `plot_convex_hulled` and `plot_convex_hulled_MF_dev` is
    used to specify the colors for different conditions or regions in the plots. The colors are assigned
    based on the conditions or regions being plotted.
    pre_cond
        The `pre_cond` parameter in the `plot_convex_hulled` and `plot_convex_hulled_MF_dev` functions is
    used to specify a prefix condition that can be added to the label of the plotted data points. If
    `pre_cond` is not `None`, it will be
    savefile
        The `savefile` parameter in both `plot_convex_hulled` and `plot_convex_hulled_MF_dev` functions is
    a string that specifies the file path where the generated plot will be saved. It should include the
    file name and extension (e.g., "plot.png", "
    
    '''
    xmax, xmin = np.amax(X_reduced[:, 0]), np.amin(X_reduced[:, 0])
    ymax, ymin = np.amax(X_reduced[:, 1]), np.amin(X_reduced[:, 1])

    fig, ax = plt.subplots(dpi=300)
    fig.set_size_inches(12, 10)

    for labs in background_regions:
        for conds in conditions:
            inds = np.where(
                (np.array(brain_conds) == conds) * (np.array(brain_labels) == labs)
            )[0]
            # ax.scatter(X_reduced[inds][:,0], X_reduced[inds][:,1], s=30,
            #           c='lightgrey', marker='o', lw=1.5, alpha=0.4, rasterized=True)

            distances = pdist(X_reduced[inds])
            distances = squareform(distances)
            graph = (distances <= 1.0).astype("int")
            graph = csr_matrix(graph)
            n_components, labels = connected_components(
                csgraph=graph, directed=False, return_labels=True
            )

            inds = inds[np.where(labels == 0)[0]]
            hull = ConvexHull(X_reduced[inds])
            for simplex in hull.simplices:
                plt.plot(
                    X_reduced[inds][simplex, 0],
                    X_reduced[inds][simplex, 1],
                    color=colors[conds],
                    ls="--",
                    lw=2,
                )

    for cond in conditions:
        for labs in foreground_regions:
            inds = np.where(
                (np.array(brain_conds) == cond) * (np.array(brain_labels) == labs)
            )[0]
            if pre_cond != None:
                label = "%s, %s %s" % (labs, pre_cond, cond)
            else:
                label = "%s, %s" % (labs, cond)
            ax.scatter(
                X_reduced[:, 0][inds],
                X_reduced[:, 1][inds],
                s=sizes[cond],
                c=colors[cond],
                marker="o",
                edgecolor="k",
                lw=0.4,
                label=label,
                rasterized=True,
            )

    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)

    ax.set_xlim(left=xmin * (1.1), right=xmax * (1.1))
    ax.set_ylim(bottom=ymin * (1.1), top=ymax * (1.1))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ax.legend(loc="lower right", fontsize=12)

    plt.savefig(savefile, bbox_inches="tight", dpi=300)


    
def plot_convex_hulled_MF_dev(
    X_reduced,
    foreground_region,
    brain_conds,
    brain_labels,
    brain_genders,
    conditions,
    sizes,
    colors_M,
    colors_F,
    pre_cond,
    savefile,
):
    '''The function `plot_convex_hulled_MF_dev` generates plots of convex hulls for different conditions
    and genders in a given dataset.
    
    Parameters
    ----------
    X_reduced
        It seems like you were about to provide information about the `X_reduced` parameter in the
    `plot_convex_hulled_MF_dev` function. Please go ahead and provide the details or let me know if you
    need any specific assistance with this parameter or any other part of the code.
    foreground_region
        The `foreground_region` parameter in the `plot_convex_hulled_MF_dev` function represents the region
    of interest that will be highlighted in the plot. It is used to filter data points based on this
    region for visualization purposes.
    brain_conds
        `brain_conds` appears to represent the conditions related to brain data. It is used in the function
    `plot_convex_hulled_MF_dev` to filter data based on specific conditions.
    brain_labels
        The `brain_labels` parameter likely represents the labels associated with different regions of the
    brain. These labels could indicate specific brain regions or regions of interest within the brain.
    The function `plot_convex_hulled_MF_dev` appears to use these labels to identify and visualize
    specific brain regions in relation to
    brain_genders
        The `brain_genders` parameter in the `plot_convex_hulled_MF_dev` function likely represents the
    genders associated with the data points in the brain dataset. It is used to filter and differentiate
    data points based on gender when plotting the convex hulls and scatter points in the visualization.
    conditions
        The `conditions` parameter in the `plot_convex_hulled_MF_dev` function represents a list of
    conditions that are used to filter data during the plotting process. These conditions are used to
    select specific data points based on certain criteria for visualization.
    sizes
        The `sizes` parameter in the `plot_convex_hulled_MF_dev` function likely represents the sizes of
    the data points to be plotted for different conditions. It seems to be a dictionary where the keys
    correspond to different conditions and the values represent the size of the data points for each
    condition.
    colors_M
        The `colors_M` parameter likely contains color values for different conditions associated with male
    gender. These colors are used for plotting data points and convex hulls in the function
    `plot_convex_hulled_MF_dev`. Each condition in the `conditions` list probably has a corresponding
    color specified in the `
    colors_F
        `colors_F` is a dictionary that likely contains color values for different conditions associated
    with female gender. It is used in the `plot_convex_hulled_MF_dev` function to determine the color of
    data points based on the condition and gender being plotted.
    pre_cond
        The `pre_cond` parameter in the `plot_convex_hulled_MF_dev` function seems to be used for
    specifying a prefix condition label in the plot legend. It is used to create a label that includes
    the foreground region, the prefix condition, the condition, and the foreground gender.
    savefile
        The `savefile` parameter is a string that specifies the path and filename prefix for saving the
    generated plots. It will be used to save the plots as PDF files with the gender information appended
    to the filename.
    
    '''
    xmax, xmin = np.amax(X_reduced[:, 0]), np.amin(X_reduced[:, 0])
    ymax, ymin = np.amax(X_reduced[:, 1]), np.amin(X_reduced[:, 1])

    marker = {}
    marker["M"] = "o"
    marker["F"] = "^"

    linestyle = {}
    linestyle["M"] = "--"
    linestyle["F"] = "dotted"

    size_offset = {}
    size_offset["M"] = 0
    size_offset["F"] = 20

    def plot_convex_hulled_sex_dev(foreground_gender, background_gender):
        fig, ax = plt.subplots(dpi=300)
        fig.set_size_inches(12, 10)

        for conds in conditions:
            inds = np.where(
                (np.array(brain_conds) == conds)
                * (np.array(brain_labels) == foreground_region)
                * (np.array(brain_genders) == background_gender)
            )[0]

            distances = pdist(X_reduced[inds])
            distances = squareform(distances)
            graph = (distances <= 0.8).astype("int")
            graph = csr_matrix(graph)
            n_components, labels = connected_components(
                csgraph=graph, directed=False, return_labels=True
            )

            largest_component = np.argmax(
                [len(np.where(labels == i)[0]) for i in np.unique(labels)]
            )
            inds = inds[np.where(labels == largest_component)[0]]
            hull = ConvexHull(X_reduced[inds])

            if background_gender == "M":
                color = colors_M[conds]
            elif background_gender == "F":
                color = colors_F[conds]
            else:
                color = "darkgrey"

            for simplex in hull.simplices:
                plt.plot(
                    X_reduced[inds][simplex, 0],
                    X_reduced[inds][simplex, 1],
                    color=color,
                    ls=linestyle[background_gender],
                    lw=2,
                )

        for conds in conditions:
            inds = np.where(
                (np.array(brain_conds) == conds)
                * (np.array(brain_labels) == foreground_region)
                * (np.array(brain_genders) == foreground_gender)
            )[0]
            if pre_cond != None:
                label = "%s, %s %s (%s)" % (
                    foreground_region,
                    pre_cond,
                    conds,
                    foreground_gender,
                )
            else:
                label = "%s, %s (%s)" % (foreground_region, conds, foreground_gender)

            if foreground_gender == "M":
                color = colors_M[conds]
            elif foreground_gender == "F":
                color = colors_F[conds]
            else:
                color = "darkgrey"

            ax.scatter(
                X_reduced[:, 0][inds],
                X_reduced[:, 1][inds],
                s=sizes[conds] + size_offset[foreground_gender],
                c=color,
                marker=marker[foreground_gender],
                edgecolor="k",
                lw=0.4,
                label=label,
                rasterized=True,
            )

        ax.set_xlabel("UMAP 1", fontsize=14)
        ax.set_ylabel("UMAP 2", fontsize=14)

        ax.set_xlim(left=xmin * (1.1), right=xmax * (1.1))
        ax.set_ylim(bottom=ymin * (1.1), top=ymax * (1.1))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        ax.legend(loc="lower right", fontsize=12)

        plt.savefig(
            savefile + "_%s.pdf" % foreground_gender, bbox_inches="tight", dpi=300
        )

    plot_convex_hulled_sex_dev("M", "F")
    plot_convex_hulled_sex_dev("F", "M")


    
def plot_convex_hulled_MF_deg_spatial(
    X_reduced,
    foreground_region,
    brain_conds,
    brain_labels,
    brain_genders,
    conditions,
    sizes,
    colors_M,
    colors_F,
    pre_cond,
    savefile,
):
    '''This function plots convex hulls for different conditions and genders in a spatial plot.
    
    Parameters
    ----------
    X_reduced
        X_reduced is a 2D array containing the reduced data points after dimensionality reduction,
    typically used for visualization purposes.
    foreground_region
        The `foreground_region` parameter in the `plot_convex_hulled_MF_deg_spatial` function represents
    the region of interest that will be highlighted in the plot. It is used to specify which region of
    the brain data should be emphasized in the visualization.
    brain_conds
        The `brain_conds` parameter in the `plot_convex_hulled_MF_deg_spatial` function represents the
    conditions associated with brain data. These conditions could be different experimental conditions
    or categories under which the brain data is classified. The function uses these conditions to
    differentiate and visualize different subsets of the
    brain_labels
        The `brain_labels` parameter in the `plot_convex_hulled_MF_deg_spatial` function represents the
    labels associated with each data point in the `X_reduced` dataset. These labels are used to identify
    different regions or groups within the data for visualization and analysis purposes. In the function
    brain_genders
        `brain_genders` appears to be a parameter that stores the gender information related to the brain
    data being analyzed. It likely contains the gender labels associated with each data point in the
    `X_reduced` dataset. The function `plot_convex_hulled_MF_deg_spatial` seems to use
    conditions
        The `conditions` parameter in the `plot_convex_hulled_MF_deg_spatial` function represents a list of
    conditions that are used to filter data points during the plotting process. These conditions are
    used to identify specific subsets of data for visualization based on the values present in the
    `brain_conds
    sizes
        The `sizes` parameter in the `plot_convex_hulled_MF_deg_spatial` function likely represents the
    sizes of the markers used in the scatter plot for different conditions. It is a dictionary where the
    keys are conditions and the values are the sizes to be used for markers corresponding to those
    conditions
    colors_M
        The `colors_M` parameter likely contains color values for different conditions associated with male
    gender. These colors are used for plotting data points in the visualization based on the conditions
    and gender specified in the function.
    colors_F
        The `colors_F` parameter likely contains color values for different conditions associated with
    female gender. These colors are used for plotting data points and convex hulls in the function
    `plot_convex_hulled_MF_deg_spatial`. The function seems to be visualizing spatial data based on
    gender and conditions,
    pre_cond
        The `pre_cond` parameter in the `plot_convex_hulled_MF_deg_spatial` function seems to be a variable
    that is defined but not used within the function. It is listed as one of the input parameters but is
    not referenced anywhere in the function body.
    savefile
        The `savefile` parameter in the `plot_convex_hulled_MF_deg_spatial` function is used to specify the
    file path where the generated plot will be saved. If you want to save the plot to a specific file,
    you can provide the file path as a string to the `
    
    '''
    xmax, xmin = np.amax(X_reduced[:, 0]), np.amin(X_reduced[:, 0])
    ymax, ymin = np.amax(X_reduced[:, 1]), np.amin(X_reduced[:, 1])

    marker = {}
    marker["M"] = "o"
    marker["F"] = "^"

    linestyle = {}
    linestyle["M"] = "--"
    linestyle["F"] = "dotted"

    size_offset = {}
    size_offset["M"] = 0
    size_offset["F"] = 20

    def plot_convex_hulled_sex_deg(foreground_gender, background_gender):
        fig, ax = plt.subplots(dpi=300)
        fig.set_size_inches(12, 10)

        inds = np.where(np.array(brain_labels) != foreground_region)[0]
        ax.scatter(
            X_reduced[:, 0][inds],
            X_reduced[:, 1][inds],
            s=30,
            c="lightgrey",
            marker="o",
            alpha=0.4,
            rasterized=True,
        )

        for conds in conditions:
            inds = np.where(
                (np.array(brain_conds) == conds)
                * (np.array(brain_labels) == foreground_region)
                * (np.array(brain_genders) == background_gender)
            )[0]

            distances = pdist(X_reduced[inds])
            distances = squareform(distances)
            graph = (distances <= 0.8).astype("int")
            graph = csr_matrix(graph)
            n_components, labels = connected_components(
                csgraph=graph, directed=False, return_labels=True
            )

            largest_component = np.argmax(
                [len(np.where(labels == i)[0]) for i in np.unique(labels)]
            )
            inds = inds[np.where(labels == largest_component)[0]]
            hull = ConvexHull(X_reduced[inds])

            if background_gender == "M":
                color = colors_M[conds]
            elif background_gender == "F":
                color = colors_F[conds]
            else:
                color = "darkgrey"

            for simplex in hull.simplices:
                plt.plot(
                    X_reduced[inds][simplex, 0],
                    X_reduced[inds][simplex, 1],
                    color=color,
                    ls=linestyle[background_gender],
                    lw=2,
                )

        for conds in conditions:
            inds = np.where(
                (np.array(brain_conds) == conds)
                * (np.array(brain_labels) == foreground_region)
                * (np.array(brain_genders) == foreground_gender)
            )[0]

            label = "%s, %s (%s)" % (foreground_region, conds, foreground_gender)

            if foreground_gender == "M":
                color = colors_M[conds]
            elif foreground_gender == "F":
                color = colors_F[conds]
            else:
                color = "darkgrey"

            ax.scatter(
                X_reduced[:, 0][inds],
                X_reduced[:, 1][inds],
                s=sizes[conds] + size_offset[foreground_gender],
                c=color,
                marker=marker[foreground_gender],
                edgecolor="k",
                lw=0.4,
                label=label,
                rasterized=True,
            )

        ax.set_xlabel("UMAP 1", fontsize=14)
        ax.set_ylabel("UMAP 2", fontsize=14)

        ax.set_xlim(left=xmin * (1.1), right=xmax * (1.1))
        ax.set_ylim(bottom=ymin * (1.1), top=ymax * (1.1))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        ax.legend(loc="lower right", fontsize=12)

        if savefile != "FALSE":
            plt.savefig(
                savefile + "_%s.pdf" % foreground_gender, bbox_inches="tight", dpi=300
            )

    plot_convex_hulled_sex_deg("M", "F")
    plot_convex_hulled_sex_deg("F", "M")


def quick_pyvol_UMAP(_X_umap, _info_frame, _conditions, _regions, _color):
    '''The function `quick_pyvol_UMAP` creates a 3D scatter plot using pyvol with specified conditions,
    regions, and colors.
    
    Parameters
    ----------
    _X_umap
        _X_umap is a numpy array containing the UMAP coordinates for each data point. It has shape
    (n_samples, 3) where n_samples is the number of data points and the 3 columns represent the UMAP
    dimensions.
    _info_frame
        The `_info_frame` parameter is likely a DataFrame containing information about different regions
    and models. It seems to have columns like 'Region' and 'Model' that are used to filter data points
    for visualization in the UMAP plot.
    _conditions
        _conditions is a string that is used to filter the data based on a specific condition or criteria.
    It is used in the function to select data points that meet the specified condition.
    _regions
        The `_regions` parameter in the `quick_pyvol_UMAP` function is a list containing the regions for
    which you want to plot data points in the 3D UMAP visualization. Each region in the list corresponds
    to a specific subset of data points that will be highlighted in the plot based
    _color
        The `_color` parameter is a dictionary that maps each region to a specific color. This allows you
    to customize the color of each region when plotting the UMAP data points.
    
    '''
    ipv.figure()
    ipv.scatter(
        _X_umap[:, 0],
        _X_umap[:, 1],
        _X_umap[:, 2],
        size=0.1,
        color="lightgrey",
        marker="sphere",
    )
    for region in _regions:
        _inds = np.where(
            (_info_frame.Region == region)
            * (_info_frame.Model.str.contains(_conditions))
        )[0]
        if len(_inds) > 0:
            ipv.scatter(
                _X_umap[:, 0][_inds],
                _X_umap[:, 1][_inds],
                _X_umap[:, 2][_inds],
                size=0.5,
                color=_color[region],
                marker="sphere",
                label=region,
            )
    ipv.show()
    

    
def _get_condition(_model):
    '''The function `_get_condition` returns a specific condition based on the input `_model`.
    
    Parameters
    ----------
    _model
        The `_model` parameter in the `_get_condition` function represents different genetic or
    environmental conditions in a research study. The function checks the value of `_model` and returns
    a corresponding condition category. If `_model` is "Cx3cr1_het" or "Light_rearing", the condition
    
    Returns
    -------
        The function `_get_condition` returns a string indicating the condition based on the input
    `_model`. If the `_model` is "Cx3cr1_het" or "Light_rearing", it returns "Development". If the
    `_model` is "rd1" or "rd10", it returns "Degeneration".
    
    '''
    if _model in ["Cx3cr1_het", "Light_rearing"]:
        return "Development"
    if _model in ["rd1", "rd10"]:
        return "Degeneration"


    
def _pyvol_scatter_UMAP(_X_umap, _info_frame, _model, _region, _cmaps):
    '''This function creates a 3D scatter plot using UMAP coordinates with different colors based on
    timepoints and regions.
    
    Parameters
    ----------
    _X_umap
        _X_umap is a NumPy array containing the UMAP embeddings of the data points. It has shape
    (n_samples, 3) where n_samples is the number of data points and each row represents the coordinates
    of a data point in the UMAP space.
    _info_frame
        The `_info_frame` parameter in the `_pyvol_scatter_UMAP` function is likely a DataFrame containing
    information about regions, models, timepoints, and conditions. It is used to filter data based on
    the specified region and model in the function to visualize the UMAP scatter plot.
    _model
        The `_model` parameter in the `_pyvol_scatter_UMAP` function is used to specify a particular model
    for which data will be plotted on a 3D scatter plot. It is used to filter the data based on the
    model specified and then visualize it in the plot.
    _region
        The `_region` parameter in the `_pyvol_scatter_UMAP` function is used to specify a particular
    region for which data will be plotted in a 3D scatter plot. This function seems to be creating a
    visualization using the `ipyvolume` library, where data points are scattered in
    _cmaps
        The `_cmaps` parameter in the `_pyvol_scatter_UMAP` function seems to be a dictionary containing
    colormaps for different conditions and regions. It is used to assign colors to data points based on
    the condition and region they belong to.
    
    '''
    ipv.scatter(
        _X_umap[:, 0],
        _X_umap[:, 1],
        _X_umap[:, 2],
        size=1,
        color="lightgrey",
        marker="sphere",
        alpha=0.3,
    )

    query_frame = _info_frame.query("Region == @_region & Model == @_model")
    timepoints = query_frame.Time.unique()
    time_vals = [
        int(_time.split("_")[0][1:]) if _time.split("_")[0][0] == "P" else 30
        for _time in timepoints
    ]
    _cond = _get_condition(_model)

    for _t in np.arange(len(timepoints)):
        _time = timepoints[_t]
        _inds = query_frame.query("Time == @_time").index.values

        color = [
            _cmaps[_cond][_region]((time_vals[_t] + 5) / (max(time_vals) * 1.01))
        ] * len(_inds)

        if len(_inds) > 0:
            ipv.scatter(
                _X_umap[:, 0][_inds],
                _X_umap[:, 1][_inds],
                _X_umap[:, 2][_inds],
                size=2,
                color=color,
                marker="sphere",
                label=_time,
                alpha=0.7,
            )


def quick_pyvol_UMAP_trajectory(_X_umap, _info_frame, _model, _regions, _cmaps):
    '''This function creates a 3D interactive visualization using UMAP embeddings for different regions.
    
    Parameters
    ----------
    _X_umap
        It looks like the function `quick_pyvol_UMAP_trajectory` is meant to create a 3D visualization
    using the ipyvolume library. The function iterates over a list of regions and calls
    `_pyvol_scatter_UMAP` function for each region to scatter plot the UMAP
    _info_frame
        It looks like you were about to provide information about the `_info_frame` parameter, but the
    information is missing. Could you please provide more details or let me know if you need help with
    something specific related to the `_info_frame` parameter?
    _model
        It looks like the `_model` parameter in the `quick_pyvol_UMAP_trajectory` function is used within
    the `_pyvol_scatter_UMAP` function to visualize data points in a specific region on a 3D plot. The
    `_model` parameter likely contains information or settings related
    _regions
        It looks like the `_regions` parameter is used as a list of regions that will be iterated over in
    the `quick_pyvol_UMAP_trajectory` function. Each region in the list will be passed to the
    `_pyvol_scatter_UMAP` function along with other parameters such as
    _cmaps
        It looks like you haven't provided the definition for the `_cmaps` parameter in your function
    `quick_pyvol_UMAP_trajectory`. The `_cmaps` parameter likely refers to the color maps that you want
    to use for plotting the UMAP trajectory in your visualization.
    
    '''
    ipv.figure()
    ipv.style.box_off()
    for _region in _regions:
        _pyvol_scatter_UMAP(_X_umap, _info_frame, _model, _region, _cmaps)
    ipv.show()
    
    
    
        
# NEW FUNCTIONS FOR PLOTTING WITH IPYVOLUME
def scatterplot_3D(_coordinates, _size, _color, _description, _alpha):
    '''The `scatterplot_3D` function creates a 3D scatter plot using ipyvolume with specified
        coordinates, size, color, description, and transparency.
        
        Parameters
        ----------
        _coordinates
            The `_coordinates` parameter in the `scatterplot_3D` function is expected to be a 3D NumPy
        array containing the coordinates of the points you want to plot in the 3D space. Each row of the
        array represents the coordinates of a single point in the format [x
        _size
            The `_size` parameter in the `scatterplot_3D` function is used to specify the size of the
        markers (spheres) in the 3D scatter plot. It controls the size of the spheres representing the
        data points in the plot. You can adjust this parameter to make the markers
        _color
            The `_color` parameter in the `scatterplot_3D` function is used to specify the color of the
        markers in the 3D scatter plot. You can provide a single color or an array of colors to
        customize the appearance of the markers in the plot.
        _description
            The `_description` parameter in the `scatterplot_3D` function is used to provide a description
        or label for the scatter plot being created. This description can help users understand the
        purpose or content of the plot at a glance. It is typically a string that describes the data or
        the visualization being
        _alpha
            The `_alpha` parameter in the `scatterplot_3D` function controls the transparency level of the
        plotted points in the 3D scatter plot. A value of 0 means completely transparent, while a value
        of 1 means completely opaque. You can adjust this parameter to control how see-through
        
        '''
    ipv.scatter(
        _coordinates[:, 0],
        _coordinates[:, 1],
        _coordinates[:, 2],
        size=_size,
        color=_color,
        marker="sphere",
        description=_description,
        alpha=_alpha,
    )

def scatterplot_3D_all(_coordinates):
    '''The function `scatterplot_3D_all` creates a 3D scatter plot with all points using specified
    parameters.
    
    Parameters
    ----------
    _coordinates
        The `_coordinates` parameter is a list of 3D coordinates representing points in a 3D space. Each
    coordinate should be a tuple or list containing three values (x, y, z) representing the position of
    a point in the 3D space.
    
    '''
    scatterplot_3D(_coordinates, 1, "lightgrey", "All points", 0.3)

    
def scatterplot_3D_conditions(
    _coordinates, _morpho_info, _conditions, _colormap, _label_prefix=None
):
    '''The function `scatterplot_3D_conditions` generates a 3D scatter plot based on conditions and
    colormap information provided.
    
    Parameters
    ----------
    _coordinates
        The `_coordinates` parameter is expected to be a numpy array containing the coordinates for
    plotting the 3D scatterplot. It should have a shape of (n, 3) where n is the number of data points
    and 3 represents the x, y, and z coordinates respectively.
    _morpho_info
        _morpho_info: This parameter is a DataFrame containing morphological information data.
    _conditions
        The `_conditions` parameter in the `scatterplot_3D_conditions` function represents the conditions
    based on which the scatterplot will be generated. These conditions are used to filter the data in
    `_morpho_info` to select specific data points for plotting. The function iterates over different
    conditions specified in
    _colormap
        _colormap is a DataFrame containing information about the colors to be used for different
    conditions in the scatterplot. It includes columns such as 'Color_type', 'Color', and
    'GradientLimits' that specify how the colors should be applied based on the conditions provided.
    _label_prefix
        The `_label_prefix` parameter is a string that serves as a prefix for the labels in the
    scatterplot. If provided, it will be added to the beginning of each label. If not provided, it will
    default to an empty string.
    
    '''
    if _label_prefix is not None:
        _label_prefix = _label_prefix + "_"
    else:
        _label_prefix = ""

    for _idx_conds in range(len(_colormap)):
        conds = _colormap.loc[_idx_conds, _conditions]
        color_type = _colormap.iloc[_idx_conds].Color_type
        color = _colormap.iloc[_idx_conds].Color
        limits = _colormap.iloc[_idx_conds].GradientLimits

        query_frame = _morpho_info.copy()
        for _conds in _conditions:
            if conds[_conds] in _morpho_info[_conds].unique():
                query_frame = query_frame.loc[query_frame[_conds] == conds[_conds]]
        time_inds = query_frame.index.values

        if color_type == "solid":
            _c = color[0]

        elif color_type in ["gradient_mod", "gradient_pre"]:
            if color_type == "gradient_mod":
                if len(color) == 1:
                    color = ("white", color[0])
                _colors = mpl.colors.colors.LinearSegmentedColormap.from_list("", color)
            # else:
            _colors = color[0]

            assert (
                "Time" in query_frame.columns
            ), "Using color gradients. Time must be a component of `morpho_info`. Check your data!"

            norm = plt.Normalize(limits[0], limits[1])
            time_vals = np.array(query_frame.Time.str.replace("P", "")).astype("int")
            _c = cm.get_cmap(_colors)(norm(time_vals))

            assert len(_coordinates[time_inds, 0]) == len(
                time_inds
            ), "Something went wrong with identifying the indices containing the conditions"

        else:
            raise ValueError(
                "`Color_type` must either be `solid`, `gradient_mod` or `gradient_pre`"
            )

        if len(time_inds) > 0:
            scatterplot_3D(
                _coordinates[time_inds],
                2,
                _c,
                "%s%s" % (_label_prefix, "_".join(conds.values)),
                0.7,
            )
            
                
def scatterplot_3D_spectrum(coordinates, morpho_info, conditions, colormap, savefile=None):
    '''The function `scatterplot_3D_spectrum` generates a 3D scatter plot of coordinates with
        morphological information colored by conditions using a specified colormap.
        
        Parameters
        ----------
        coordinates
            Coordinates are typically the data points or locations in a 3D space where each point
        represents a data instance or sample. In this context, the coordinates likely refer to the
        UMAP (Uniform Manifold Approximation and Projection) coordinates of the data points in a 3D
        space. These coordinates are
        morpho_info
            The `morpho_info` parameter likely contains information about the morphological
        characteristics of the data points in the scatterplot. This information could include details
        such as cell morphology features, measurements, or any other relevant data that helps in
        visualizing and analyzing the data points in the scatterplot.
        conditions
            The `conditions` parameter likely refers to the different conditions or categories that the
        data points in the scatterplot represent. These conditions could be different experimental
        groups, treatments, or any other categorical variable that you want to visualize in the 3D
        scatterplot. The function `scatterplot_3D_spectrum
        colormap
            The `colormap` parameter in the `scatterplot_3D_spectrum` function is used to specify the
        color mapping scheme for visualizing the data points in the 3D scatter plot. It allows you to
        assign different colors to data points based on a particular variable or condition, making it
        easier
        savefile
            The `savefile` parameter in the `scatterplot_3D_spectrum` function is used to specify the
        name of the file where the 3D scatterplot will be saved as an HTML file. If you provide a
        value for `savefile`, the function will save the plot as an HTML
        
        '''
    ipv.figure(width=1920, height=1080)
    ipv.style.box_off()

    scatterplot_3D_all(coordinates)
    scatterplot_3D_conditions(coordinates, morpho_info, conditions, colormap)

    ipv.xyzlabel("UMAP 1", "UMAP 2", "UMAP 3")

    if savefile is not None:
        ipv.save(
            "%s.html"%savefile,
            title="Morphological spectrum",
            offline=False,
        )

    ipv.show()