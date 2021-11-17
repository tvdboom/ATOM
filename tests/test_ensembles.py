# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for ensembles.py

"""

# Standard packages
import pytest
import numpy as np

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.utils import check_scaling
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg


# Voting =========================================================== >>


# Stacking ========================================================= >>

