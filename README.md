# morphOMICs

`morphOMICs` is a Python package containing tools for analyzing microglia morphology using a topological data analysis approach. Note that this algorithm is designed not only for microglia applications but also for any dynamic branching structures across natural sciences.

- [Overview](#overview)
- [Required Dependencies](#required-dependencies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)

# Overview
`morphOMICs` is a topological data analysis approach which combines the Topological Morphology Descriptor (TMD) with bootstrapping approach, dimensionality reduction strategies to visualize microglial morphological signatures and their relationships across different biological conditions.


# Required Dependencies
Python : <= 3.10

numpy : 1.8.1+, scipy : 0.13.3+, pickle : 4.0+, enum34 : 1.0.4+, scikit-learn : 0.19.1+, tomli: 2.0.1+, matplotlib : 3.2.0+, ipyvolume: 0.6.1+, umap-learn : 0.3.10+, morphon: 0.0.8+, pylmeasure: 0.2.0+, fa2_modified

# Installation Guide

You need Python 3.9 or 3.10 to run this package.
``` console
conda create -n morphology python=3.9
conda activate morphology
pip install morphomics
```


# Usage
To run a typical morphOMICs pipeline, create a .toml parameter file (see examples).
The parameter file is build such that it modularizes the steps required to generate the phenotypic spectrum.
Once you have completed filling up the necessary information in the parameter file, you can use the `examples\run.ipynb` file to have an idea on how to run this program.
