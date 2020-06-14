# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: File to run before starting the unit tests.

"""
# Import packages
import os
import glob
from .utils import FILE_DIR


# Remove previously created files
files = glob.glob(FILE_DIR + '*')
for f in files:
    os.remove(f)
