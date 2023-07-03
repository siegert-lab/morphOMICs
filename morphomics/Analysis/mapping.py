"""
morphomics : linear regression and mapping tools

Author: Ryan Cubero
"""
import numpy as np
import pickle as pkl

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from morphomics.utils import load_obj

mapping_methods = {
    "mean": lambda arr: np.mean(arr),
    "median": lambda arr: np.median(arr),
    "std": lambda arr: np.std(arr),
}


def get_reference_atlas(location, region):
    complete_data = load_obj("%s/ReferenceAtlas_%s" % (location, region))
    return complete_data["reference_palantir"], complete_data["conditions"]


def get_model(location, region):
    complete_data = load_obj("%s/ReferenceAtlas_%s" % (location, region))
    return complete_data["reg_x"], complete_data["reg_y"]


def calculate_hvg(train_data, n_hvg):
    _std = np.std(train_data, axis=0)
    ph_hvg = np.where(_std >= np.sort(_std)[-int(n_hvg)])[0]

    return ph_hvg


def train_model(X_train, y_train):
    reg_x = LinearRegression().fit(X_train, y_train[:, 0])
    reg_y = LinearRegression().fit(X_train, y_train[:, 1])

    return reg_x, reg_y


def map_coordinates(test_ph, reg_x, reg_y, method="median", return_std=True):
    x_pred = reg_x.predict(test_ph)
    y_pred = reg_y.predict(test_ph)

    data = {}
    data["x_pred"] = mapping_methods[method](x_pred)
    data["y_pred"] = mapping_methods[method](y_pred)

    if return_std:
        data["x_std"] = mapping_methods["std"](x_pred)
        data["y_std"] = mapping_methods["std"](y_pred)

    return data
