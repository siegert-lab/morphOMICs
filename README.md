# morphOMICs

`morphOMICs` is a Python package containing tools for analyzing microglia morphology using a topological data analysis approach. Note that this algorithm is designed not only for microglia applications but also for any dynamic branching structures across natural sciences.

- [morphOMICs](#morphomics)
- [Overview](#overview)
- [Required Dependencies](#required-dependencies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)

# Overview
`morphOMICs` is a topological data analysis approach which combines the Topological Morphology Descriptor (TMD) with bootstrapping approach, dimensionality reduction strategies to visualize microglial morphological signatures and their relationships across different biological conditions.


# Required Dependencies

This project relies on a range of scientific computing and machine learning libraries to facilitate data analysis, visualization, and modeling. Key dependencies include NumPy, Pandas, and SciPy for numerical computations, scikit-learn for machine learning, and torch alongside torch-geometric for deep learning applications. Additionally, Matplotlib and UMAP-learn support data visualization and dimensionality reduction, while NetworkX aids in graph-based analyses. The project also integrates H5Py for handling HDF5 files and WandB for experiment tracking. Ensure all dependencies are installed to guarantee full functionality.

# Installation Guide

### Clone the Repository:**  
   `git clone git@github.com:ThomasNgl/morphOMICs.git`  
   `cd morphOMICs`

You need Python 3.9 or 3.10 to run this package.
You can install morphOMICs using either Conda or pip. Follow the steps below based on your preferred package manager.

### Using Conda
2. **Create and Activate the Conda Environment:**  
   `conda env create -f environment.yml`  
   `conda activate morphomics`

3. **Install the Package:**  
   `pip install -e .`

### Using pip
2. **Create and Activate the Virtual Environment:**  
   `python -m venv morphomics`  
   On macOS/Linux:  
   `source morphomics/bin/activate`  
   On Windows:  
   `morphomics\Scripts\activate`

3. **Install the Required Packages:**  
   `pip install -r requirements.txt`  
   `pip install -e .`

# Usage
To run a typical morphOMICs pipeline, create a .toml parameter file (see examples).
The parameter file is build such that it modularizes the steps required to generate the phenotypic spectrum.
Once you have completed filling up the necessary information in the parameter file, you can use the `examples\run.ipynb` file to have an idea on how to run this program.
