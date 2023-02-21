# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for api.py

"""

from sklearn.linear_model import HuberRegressor

from atom import ATOMClassifier, ATOMModel, ATOMRegressor

from .conftest import X_bin, X_reg, y_bin, y_reg


def test_atommodel():
    """Assert that it returns an estimator that works with atom."""
    model = ATOMModel(
        estimator=(huber := HuberRegressor()),
        name="huber1",
        acronym="huber",
        needs_scaling=True,
    )

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(model)
    assert model is not huber  # Is cloned
    assert model.name == "huber1"
    assert model.acronym == "huber"
    assert model.needs_scaling is True
    assert model.native_multioutput is False
    assert model.has_validation is None


def test_atomclassifier():
    """Assert that the goal is set correctly for ATOMClassifier."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.goal == "class"


def test_atomregressor():
    """Assert that the goal is set correctly for ATOMRegressor."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.goal == "reg"
