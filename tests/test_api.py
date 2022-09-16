# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for api.py

"""

import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import HuberRegressor

from atom import ATOMClassifier, ATOMLoader, ATOMModel, ATOMRegressor
from atom.data_cleaning import Imputer
from atom.training import DirectClassifier
from atom.utils import merge

from .conftest import X_bin, X_reg, y_bin, y_reg


# Test ATOMModel =================================================== >>

def test_name():
    """Assert that the name is attached to the estimator."""
    model = ATOMModel(HuberRegressor(), acronym="huber")
    assert model.acronym == "huber"


def test_needs_scaling():
    """Assert that the needs_scaling is attached to the estimator."""
    model = ATOMModel(HuberRegressor(), acronym="huber", needs_scaling=True)
    assert model.needs_scaling is True


# Test ATOMLoader ================================================== >>

def test_load():
    """Assert that a trainer is loaded correctly."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.save("trainer")

    trainer2 = ATOMLoader("trainer")
    assert trainer2.__class__.__name__ == "DirectClassifier"


def test_load_data_with_no_atom():
    """Assert that an error is raised when data is provided without atom."""
    Imputer().save("imputer")
    pytest.raises(TypeError, ATOMLoader, "imputer", data=(X_bin,))


def test_load_already_contains_data():
    """Assert that an error is raised when data is provided without needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save("atom", save_data=True)
    pytest.raises(ValueError, ATOMLoader, "atom", data=(X_bin,))


def test_data():
    """Assert that data can be loaded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save("atom", save_data=False)

    atom2 = ATOMLoader("atom", data=(X_bin, y_bin))
    assert_frame_equal(atom2.dataset, atom.dataset, check_dtype=False)


def test_load_ignores_n_rows_parameter():
    """Assert that n_rows is not used when transform_data=False."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.6, random_state=1)
    atom.save("atom", save_data=False)

    atom2 = ATOMLoader("atom", data=(X_bin, y_bin), transform_data=False)
    assert len(atom2.dataset) == len(X_bin)


def test_transform_data():
    """Assert that the data is transformed correctly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale(columns=slice(3, 10))
    atom.apply(np.exp, columns=2)
    atom.feature_generation(strategy="dfs", n_features=5)
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=10)
    atom.save("atom", save_data=False)

    atom2 = ATOMLoader("atom", data=(X_bin, y_bin), transform_data=True)
    assert atom2.dataset.shape == atom.dataset.shape

    atom3 = ATOMLoader("atom", data=(X_bin, y_bin), transform_data=False)
    assert atom3.dataset.shape == merge(X_bin, y_bin).shape


def test_transform_data_multiple_branches():
    """Assert that the data is transformed with multiple branches."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.prune()
    atom.branch = "b2"
    atom.balance()
    atom.feature_generation(strategy="dfs", n_features=5)
    atom.branch = "b3"
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=20)
    atom.save("atom_2", save_data=False)

    atom2 = ATOMLoader("atom_2", data=(X_bin, y_bin), transform_data=True)
    for branch in atom._branches:
        assert_frame_equal(
            left=atom2._branches[branch]._data,
            right=atom._branches[branch]._data,
            check_dtype=False,
        )


# Test ATOMClassifier ============================================== >>

def test_goal_ATOMClassifier():
    """Assert that the goal is set correctly for ATOMClassifier."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.goal == "class"


# Test ATOMRegressor =============================================== >>

def test_goal_ATOMRegressor():
    """Assert that the goal is set correctly for ATOMRegressor."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.goal == "reg"
