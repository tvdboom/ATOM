# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Run before starting the unit tests. Removes all
             existing testing files (in "files" directory).

"""

# Standard packages
import os
import glob
from .utils import FILE_DIR


# Remove previously created files
for f in glob.glob(FILE_DIR + "*"):
    os.remove(f)
