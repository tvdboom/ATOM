"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Import API and version, and set configuration.

"""

import sklearn

from atom._show_versions import show_versions
from atom.api import ATOMClassifier, ATOMForecaster, ATOMModel, ATOMRegressor
from atom.utils.constants import __version__


sklearn.set_config(transform_output="pandas")
