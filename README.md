# morphOMICs

`morphOMICs` is a Python package containing tools for analyzing microglia morphology using a topological data analysis approach. Note that this algorithm is designed not only for microglia applications but also for any dynamic branching structures across natural sciences.

- [Overview](#overview)
- [Required Dependencies](#required-dependencies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)

# Overview
`morphOMICs` is a topological data analysis approach which combines the Topological Morphology Descriptor (TMD) with bootstrapping approach, dimensionality reduction strategies to visualize microglial morphological signatures and their relationships across different biological conditions.


# Required Dependencies
Python : 3.9

numpy : 1.8.1+, scipy : 0.13.3+, pickle : 4.0+, enum34 : 1.0.4+, scikit-learn : 0.19.1+, tomli: 2.0.1+, matplotlib : 3.2.0+, ipyvolume: 0.6.1+

Additional dependencies:
umap-learn : 0.3.10+, morphon: 0.0.8+, pylmeasure: 0.2.0+, fa2 (https://github.com/AminAlam/forceatlas2)

# Installation Guide

## Using PyPi
```
conda create -n morphology python=3.9
conda activate morphology
pip install morphomics
```

## Using Source Code
```
conda create -n morphology python=3.9
conda activate morphology

git clone https://github.com/siegert-lab/morphOMICs.git
cd morphOMICs
pip install -e .
```

# Usage
To run a typical morphOMICs pipeline, create a parameter file with filename Morphomics.Parameters.[Parameters ID].toml (see examples).
The parameter file is build such that it modularizes the steps required to generate the phenotypic spectrum.
Once you have completed filling up the necessary information in the parameter file, run 
`python3 run_morphomics.py [path-to-parameter-file]`

To get started, download a demo folder containing sample traces, a parameter file and tutorial Jupyter notebooks [HERE](https://drive.google.com/file/d/1sGE_8zHR5x-pp35lxRxgraz5f4cCb8YG/view?usp=drive_link).
Once downloaded, place the extracted folder `morphOMICs_v2_Tutorials` inside `morphomics_v2`.
On the terminal, `cd` to `morphOMICs_v2_Tutorials` and run
`python3 ../run_morphomics.py Morphomics.Parameters.1.toml`
You can also follow each steps in the protocol in Jupyter notebook in `morphOMICs_v2_Tutorials/morphOMICs_v2_Tutorial.ipynb`.
