# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for the utils module.

"""

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from atom.utils.utils import (
    ClassMap, NotFittedError, check_is_fitted, time_to_str, to_df, to_series,
)


def test_classmap_failed_initialization():
    """Assert that an error is raised when the classes do not have the key attribute."""
    with pytest.raises(ValueError, match=".*has no attribute.*"):
        ClassMap(2, 3)


def test_classmap_manipulations():
    """Assert that the ClassMap class can be manipulated."""
    cm = ClassMap(2, 3, 4, 5, 6, key="real")
    assert str(cm[[3, 4]]) == "[3, 4]"
    assert str(cm[3:5]) == "[5, 6]"
    assert str(cm[3]) == "3"
    cm[2] = 8
    assert str(cm) == "[2, 3, 8, 5, 6]"
    del cm[4]
    assert str(cm) == "[2, 3, 8, 5]"
    assert str(list(reversed(cm))) == "[5, 8, 3, 2]"
    cm += [6]
    assert str(cm) == "[2, 3, 8, 5, 6]"
    assert cm.index(3) == 1
    cm.clear()
    assert str(cm) == "[]"


def test_time_to_string():
    """Assert that the time strings are formatted properly."""
    assert time_to_str(timedelta(seconds=17).total_seconds()).startswith("17.00")
    assert time_to_str(timedelta(minutes=1, seconds=2).total_seconds()) == "01m:02s"
    assert time_to_str(timedelta(hours=3, minutes=8).total_seconds()) == "03h:08m:00s"


def test_to_pandas_with_cuml():
    """Assert that cuML objects use the to_pandas method."""
    to_df(MagicMock(spec=["to_pandas"]), columns=[0, 1])
    to_series(MagicMock(spec=["to_pandas"]))


def test_check_is_fitted_with_pandas():
    """Assert that the function works for empty pandas objects."""
    estimator = BaseEstimator()
    estimator.attr = pd.DataFrame([])
    pytest.raises(NotFittedError, check_is_fitted, estimator, attributes="attr")
    assert not check_is_fitted(estimator, exception=False, attributes="attr")
    estimator.attr = pd.Series([0, 1])
    assert check_is_fitted(estimator, attributes="attr")
