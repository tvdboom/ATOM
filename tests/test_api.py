# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for api.py

"""

# Standard packages
import pytest
from sklearn.linear_model import HuberRegressor

# Own modules
from atom import ATOMClassifier, ATOMRegressor, ATOMLoader, ATOMModel
from atom.training import DirectClassifier
from atom.data_cleaning import Imputer
from atom.utils import merge
from .utils import FILE_DIR, X_bin, y_bin, X_reg, y_reg


# Test ATOMModel =================================================== >>

def test_name():
    """Assert that the name is attached to the estimator."""
    model = ATOMModel(HuberRegressor, acronym="hub")
    assert model.acronym == "hub"


def test_fullname():
    """Assert that the fullname is attached to the estimator."""
    model = ATOMModel(HuberRegressor, acronym="hub", fullname="Hubber")
    assert model.fullname == "Hubber"


def test_needs_scaling():
    """Assert that the needs_scaling is attached to the estimator."""
    model = ATOMModel(HuberRegressor, acronym="hub", needs_scaling=True)
    assert model.needs_scaling


# Test ATOMLoader ================================================== >>

def test_load():
    """Assert that a trainer is loaded correctly."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.save(FILE_DIR + "trainer")

    trainer2 = ATOMLoader(FILE_DIR + "trainer")
    assert trainer2.__class__.__name__ == "DirectClassifier"


def test_load_data_with_no_trainer():
    """Assert that an error is raised when data is provided without a trainer."""
    Imputer().save(FILE_DIR + "imputer")
    pytest.raises(TypeError, ATOMLoader, FILE_DIR + "imputer", data=(X_bin,))


def test_load_already_contains_data():
    """Assert that an error is raised when data is provided without needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + "atom", save_data=True)
    pytest.raises(ValueError, ATOMLoader, FILE_DIR + "atom", data=(X_bin,))


def test_data():
    """Assert that data can be loaded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + "atom", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom", data=(X_bin, y_bin))
    assert atom2.dataset.equals(atom.dataset)


def test_load_ignores_n_rows_parameter():
    """Assert that n_rows is not used when transform_data=False."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.6, random_state=1)
    atom.save(FILE_DIR + "atom", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom", data=(X_bin, y_bin), transform_data=False)
    assert len(atom2.dataset) == len(X_bin)


def test_transform_data():
    """Assert that the data is transformed correctly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.prune(columns=slice(3, 10))
    atom.apply(lambda x: x+2, column="mean radius")
    atom.feature_generation(strategy="dfs", n_features=5)
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=10)
    atom.save(FILE_DIR + "atom", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom", data=(X_bin, y_bin), transform_data=True)
    assert atom2.dataset.shape == atom.dataset.shape

    atom3 = ATOMLoader(FILE_DIR + "atom", data=(X_bin, y_bin), transform_data=False)
    assert atom3.dataset.shape == merge(X_bin, y_bin).shape


def test_transform_data_multiple_branches():
    """Assert that the data is transformed with multiple branches."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.prune()
    atom.branch = "branch_2"
    atom.balance()
    atom.feature_generation(strategy="dfs", n_features=5)
    atom.branch = "branch_3"
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=20)
    atom.save(FILE_DIR + "atom_2", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom_2", data=(X_bin, y_bin), transform_data=True)
    for branch in atom._branches:
        assert atom2._branches[branch].data.equals(atom._branches[branch].data)


# Test ATOMClassifier ============================================== >>

def test_goal_ATOMClassifier():
    """Assert that the goal is set correctly for ATOMClassifier."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.goal == "classification"


# Test ATOMRegressor =============================================== >>

def test_goal_ATOMRegressor():
    """Assert that the goal is set correctly for ATOMRegressor."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.goal == "regression"
