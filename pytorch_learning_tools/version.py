from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 3 - Alpha", "Environment :: Console", "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License", "Operating System :: OS Independent", "Programming Language :: Python",
    "Topic :: Scientific/Engineering"
]

# Description should be a one-liner:
description = "pytorch_learning_tools: tools for pytorch-based machine learning projects"
# Long description will go up on the pypi page
long_description = """
pytorch_learning_tools
========
These are the modelling team's tools for making it less painful to develop,
version, test, configure, monitor, etc simple pytorch-based machine learning tools.
To get started using these components in your own software, please go to the
.. _README: https://github.com/AllenCellModeling/pytorch_learning_tools/blob/master/README.md 
License
=======
``pytorch_learning_tools`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2015--, Allen Institute for Cell Science.
"""

NAME = "pytorch_learning_tools"
MAINTAINER = "Gregory Johnson"
MAINTAINER_EMAIL = "gregj@alleninstitute.org"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/AllenCellModeling/pytorch_learning_tools"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Greg Johnson"
AUTHOR_EMAIL = "gregj@alleninstitute.org"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'pytorch_learning_tools': [pjoin('data', '*')]}
REQUIRES = ["numpy", "scipy", "torch", "matplotlib", "Ipython", "natsort", "Pillow", "h5py", "tqdm", "aicsimage"]
