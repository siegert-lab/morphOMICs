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

            if conds in FAD_conditions:
                label = "%s, 5xFAD %s (%s)" % (
                    foreground_region,
                    conds,
                    foreground_gender,
                )
            elif conds in Ckp25_conditions:
                label = "%s, Ckp25 %s (%s)" % (
                    foreground_region,
                    conds,
                    foreground_gender,
                )
            elif conds in dev_conditions:
                label = "%s, Dev %s (%s)" % (
                    foreground_region,
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

        if savefile != "FALSE":
            plt.savefig(
                savefile + "_%s.pdf" % foreground_gender, bbox_inches="tight", dpi=300
            )

    plot_convex_hulled_sex_deg("M", "F")
    plot_convex_hulled_sex_deg("F", "M")


def quick_pyvol_UMAP(_X_umap, _info_frame, _conditions, _regions, _color):
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
    if _model in ["Cx3cr1_het", "Light_rearing"]:
        return "Development"
    if _model in ["rd1", "rd10"]:
        return "Degeneration"


def _pyvol_scatter_UMAP(_X_umap, _info_frame, _model, _region, _cmaps):
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
    ipv.figure()
    ipv.style.box_off()
    for _region in _regions:
        _pyvol_scatter_UMAP(_X_umap, _info_frame, _model, _region, _cmaps)
    ipv.show()
    
    
    
# NEW FUNCTIONS FOR PLOTTING WITH IPYVOLUME
def scatterplot_3D(_coordinates, _size, _color, _description, _alpha):
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
    scatterplot_3D(_coordinates, 1, "lightgrey", "All points", 0.3)

def scatterplot_3D_conditions(
    _coordinates, _morpho_info, _conditions, _colormap, _label_prefix=None
):
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
                _colors = colors.LinearSegmentedColormap.from_list("", color)
            else:
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