# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Import API and version, and set configuration.

"""

import pandas as pd
import sklearn

from atom.api import ATOMClassifier, ATOMForecaster, ATOMModel, ATOMRegressor
from atom.utils.constants import __version__


pd.options.mode.copy_on_write = True
sklearn.set_config(transform_output="pandas")
