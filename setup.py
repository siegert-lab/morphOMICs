import os
import sys
import subprocess
from setuptools import setup
from pathlib import Path

MINIMAL_DESCRIPTION = '''morphOMICs: a python package for the topological and statistical analysis of microglia morphology (appliable to any tree structure)'''

# TORCH_VERSION = "torch==2.6.0"  # Explicitly specify the PyTorch version

# def install_package(package):
#     """Install a package using pip."""
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# def ensure_torch_installed():
#     """Ensure that the specified torch version is installed before running setup."""
#     try:
#         import torch
#         if torch.__version__ != "2.6.0":
#             print(f"Upgrading torch to {TORCH_VERSION}...")
#             install_package(TORCH_VERSION)
#     except ImportError:
#         print(f"Torch not found. Installing {TORCH_VERSION}...")
#         install_package(TORCH_VERSION)

def get_requires():
    """Read requirements.txt."""
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(requirements_file, "r") as f:
            requirements = f.read()
        return list(filter(lambda x: x != "", requirements.split()))
    except FileNotFoundError:
        return []

def read_description():
    """Read README.md and CHANGELOG.md."""
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path) as r:
            description = "\n" + r.read()
        return description
    return MINIMAL_DESCRIPTION

setup(
    name="morphomics",
    version="2.0.6",
    author='Amin Alam, Ryan Cubero, Thomas Negrello, Jens Agerberg',
    description=MINIMAL_DESCRIPTION,
    long_description=read_description(),
    long_description_content_type='text/markdown',
    install_requires=get_requires(),
    python_requires='<=3.10',
    license='GNU',
    url='https://github.com/siegert-lab/morphOMICs',
    keywords=['Morphomics', 'MicroGlia', 'UMAP', 'TDA', 
              'Topological Data Analysis', 'Microscopy', 
              'Image Analysis', 'Cell Morphology'],
    packages=['morphomics'],
    include_package_data=True,)
