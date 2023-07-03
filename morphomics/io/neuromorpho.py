"""
Python module that contains the functions
to import morphologies from NeuroMorpho.org
"""
from __future__ import print_function

import os, glob
import numpy as _np
import requests, json 

from morphomics.utils import save_obj, load_obj

# check if NeuroMorpho is up
def check_status(neuromopho_url="http://neuromorpho.org"):
    _status = False
    return _status