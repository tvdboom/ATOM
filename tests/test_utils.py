"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for the utils module.

"""

import sys
from datetime import timedelta
from unittest.mock import patch

import modin.pandas as md
import pandas as pd
import polars as pl
import pytest

from atom import show_versions
from atom.utils.utils import (
    ClassMap, time_to_str, to_df, to_series, variable_return,
)

from .conftest import X_bin, X_bin_array, y_bin, y_bin_array


# Test _show_versions ============================================== >>

def test_show_versions():
    """Assert that the show_versions function runs without errors."""
    with patch.dict("sys.modules"):
        del sys.modules["polars"]
        show_versions()


# Test utils ======================================================= >>

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


def test_to_df_None():
    """Assert that None is left as is."""
    assert to_df(None) is None


def test_to_df_numpy():
    """Assert that numpy arrays are converted to pandas objects."""
    assert isinstance(to_df(X_bin_array), pd.DataFrame)


def test_to_df_polars():
    """Assert that polars are converted to pandas objects."""
    assert isinstance(to_df(pl.from_pandas(X_bin)), pd.DataFrame)


def test_to_df_interchange():
    """Assert that interchange protocol objects are converted to pandas objects."""
    assert isinstance(to_df(md.DataFrame(X_bin)), pd.DataFrame)


def test_to_series_None():
    """Assert that None is left as is."""
    assert to_series(None) is None


def test_to_series_numpy():
    """Assert that numpy arrays are converted to series objects."""
    assert isinstance(to_series(y_bin_array), pd.Series)


def test_to_series_polars():
    """Assert that polars are converted to series objects."""
    assert isinstance(to_series(pl.from_pandas(y_bin)), pd.Series)


def test_time_to_string():
    """Assert that the time strings are formatted properly."""
    assert time_to_str(timedelta(seconds=17).total_seconds()).startswith("17.00")
    assert time_to_str(timedelta(minutes=1, seconds=2).total_seconds()) == "01m:02s"
    assert time_to_str(timedelta(hours=3, minutes=8).total_seconds()) == "03h:08m:00s"


def test_variable_return():
    """Assert that an error is raised when variable_return has both None."""
    with pytest.raises(ValueError, match=".*Both X and y can't be None.*"):
        variable_return(None, None)
