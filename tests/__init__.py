# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Run before starting the unit tests. Removes all
             existing testing files (in 'files' directory).

"""
# Import packages
import os
import glob
from .utils import FILE_DIR


# Remove previously created files
files = glob.glob(FILE_DIR + '*')
for f in files:
    os.remove(f)
