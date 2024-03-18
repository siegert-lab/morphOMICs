import os
from setuptools import setup, find_packages
from distutils.core import Extension
from pathlib import Path

MINIMAL_DESCRIPTION = '''morphOMICs: a python package for the topological and statistical  analysis of microglia morphology (appliable to any cell structure)'''

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
    name="Morphomics",
    version="2.0.7",
    author='Amin Alam, Ryan Cubero',
    description=MINIMAL_DESCRIPTION,
    long_description=read_description(),
    long_description_content_type='text/markdown',
    install_requires=get_requires(),
    python_requires='<=3.10',
    license='GNU',
    url='https://github.com/siegert-lab/morphOMICs',
    keywords=['Morhpomics', 'MicroGlia', 'UMAP', 'TDA', 'Topological Data Analysis', 'Microscopy', 'Image Analysis', 'Cell Morphology'],
    packages=['morphomics'],
    include_package_data=True,)
