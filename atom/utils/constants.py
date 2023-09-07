# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing utility constants.

"""

import numpy as np
import pandas as pd


# Current library version
__version__ = "6.0.0"

# Always considered missing values
MISSING_VALUES = [None, pd.NA, pd.NaT, np.inf, -np.inf]

# Attributes shared between atom and a dataframe
DF_ATTRS = (
    "size",
    "head",
    "tail",
    "loc",
    "iloc",
    "describe",
    "iterrows",
    "dtypes",
    "at",
    "iat",
    "memory_usage",
    "empty",
    "ndim",
)

# Default color palette (discrete color, continuous scale)
PALETTE = {
    "rgb(0, 98, 98)": "Teal",
    "rgb(56, 166, 165)": "Teal",
    "rgb(115, 175, 72)": "Greens",
    "rgb(237, 173, 8)": "Oranges",
    "rgb(225, 124, 5)": "Oranges",
    "rgb(204, 80, 62)": "OrRd",
    "rgb(148, 52, 110)": "PuRd",
    "rgb(111, 64, 112)": "Purples",
    "rgb(102, 102, 102)": "Greys",
}
