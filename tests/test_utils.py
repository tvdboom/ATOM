# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for utils.py

"""

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from atom.utils import (
    ClassMap, CustomDict, NotFittedError, check_is_fitted, time_to_str, to_df,
    to_series,
)


def test_classmap_failed_initialization():
    """Assert that the ClassMap can be initialized like any list."""
    with pytest.raises(ValueError):
        ClassMap(2, 3)


def test_classmap_manipulations():
    """Assert that the ClassMap can be initialized like any list."""
    cm = ClassMap(2, 3, 4, 5, 6, key="real")
    assert str(cm[[3, 4]]) == "[3, 4]"
    assert str(cm[3:5]) == "[5, 6]"
    assert str(cm[3]) == "3"
    cm[2] = 8
    assert str(cm) == "[2, 3, 8, 5, 6]"
    del cm[2]
    assert str(cm) == "[2, 3, 5, 6]"
    assert str(list(reversed(cm))) == "[6, 5, 3, 2]"
    cm += [6]
    assert str(cm) == "[2, 3, 5, 6, 6]"
    assert cm.index(3) == 1
    cm.clear()
    assert str(cm) == "[]"


def test_custom_dict_initialization():
    """Assert that the custom dictionary can be initialized like any dict."""
    assert str(CustomDict({"a": 0, "b": 1})) == "{'a': 0, 'b': 1}"
    assert str(CustomDict((("a", 0), ("b", 1)))) == "{'a': 0, 'b': 1}"
    assert str(CustomDict(a=0, b=1)) == "{'a': 0, 'b': 1}"


def test_custom_dict_key_request():
    """Assert that the custom dictionary key request works."""
    cd = CustomDict({"A": 0, "B": 1, "C": 2})
    assert cd["a"] == cd["A"] == cd[0] == 0
    assert cd[["a", "b"]] == CustomDict({"A": 0, "B": 1})
    assert cd[[1, 2]] == CustomDict({"B": 1, "C": 2})
    assert cd[1:3] == CustomDict({"B": 1, "C": 2})
    with pytest.raises(KeyError):
        print(cd[1.2])


def test_custom_dict_manipulations():
    """Assert that the custom dictionary accepts inserts and pops."""
    cd = CustomDict({"a": 0, "b": 1})
    assert list(cd.values()) == [0, 1]
    assert [k for k in reversed(cd)] == ["b", "a"]
    cd.insert(1, "c", 2)
    assert str(cd) == "{'a': 0, 'c': 2, 'b': 1}"
    cd.insert(0, "c", 3)
    assert str(cd) == "{'c': 3, 'a': 0, 'b': 1}"
    cd.popitem()
    assert str(cd) == "{'c': 3, 'a': 0}"
    assert cd.setdefault("c", 5) == 3
    cd.setdefault("d", 5)
    assert cd["d"] == 5
    cd.clear()
    pytest.raises(KeyError, cd.popitem)
    pytest.raises(KeyError, cd.index, "f")
    cd.update({"a": 0, "b": 1})
    assert str(cd) == "{'a': 0, 'b': 1}"
    cd.update((("c", 2), ("d", 3)))
    assert str(cd) == "{'a': 0, 'b': 1, 'c': 2, 'd': 3}"
    cd.update(e=4)
    assert str(cd) == "{'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}"
    cd.replace_key("d", "f")
    assert str(cd) == "{'a': 0, 'b': 1, 'c': 2, 'f': 3, 'e': 4}"
    cd.replace_value("f", 6)
    assert str(cd) == "{'a': 0, 'b': 1, 'c': 2, 'f': 6, 'e': 4}"
    del cd[2]
    assert "c" not in cd
    new_cd = cd.copy()
    assert new_cd is not cd


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
