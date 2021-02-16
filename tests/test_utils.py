# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for utils.py

"""

# Standard packages
import pytest
import pandas as pd
from sklearn.base import BaseEstimator

# Own modules
from atom.utils import check_is_fitted, create_acronym, NotFittedError, CustomDict


def test_check_is_fitted_with_pandas():
    """Assert the function works for empty pandas objects."""
    estimator = BaseEstimator()
    estimator.attr = pd.DataFrame([])
    pytest.raises(NotFittedError, check_is_fitted, estimator, attributes="attr")
    assert not check_is_fitted(estimator, exception=False, attributes="attr")
    estimator.attr = pd.Series([0, 1])
    assert check_is_fitted(estimator, attributes="attr")


def test_create_acronym():
    """Assert the function works as intended."""
    assert create_acronym("CustomClass") == "CC"
    assert create_acronym("Customclass") == "Customclass"


def test_custom_dict_initialization():
    """Assert the custom dictionary can be initialized like any dict."""
    assert str(CustomDict({"a": 0, "b": 1})) == "{'a': 0, 'b': 1}"
    assert str(CustomDict((("a",  0), ("b", 1)))) == "{'a': 0, 'b': 1}"
    assert str(CustomDict(a=0, b=1)) == "{'a': 0, 'b': 1}"


def test_custom_dict_keys():
    """Assert the custom dictionary only has lower case keys."""
    cd = CustomDict({"A": 0, "B": 1})
    assert str(cd) == "{'a': 0, 'b': 1}"
    assert cd["a"] == cd["A"] == cd[0] == 0


def test_custom_dict_manipulations():
    """Assert the custom dictionary accepts inserts and pops."""
    cd = CustomDict({"a": 0, "b": 1})
    cd.insert(1, "c", 2)
    assert str(cd) == "{'a': 0, 'c': 2, 'b': 1}"
    cd.insert("a", "d", 3)
    assert str(cd) == "{'d': 3, 'a': 0, 'c': 2, 'b': 1}"
    cd.popitem()
    assert str(cd) == "{'d': 3, 'a': 0, 'c': 2}"
    assert cd.setdefault("d", 5) == 3
    cd.setdefault("e", 5)
    assert cd["e"] == 5
    cd.clear()
    pytest.raises(KeyError, cd.popitem)
