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
from atom.utils import check_scaling, merge
from .utils import FILE_DIR, X_bin, y_bin, X_reg, y_reg


# Test ATOMModel =================================================== >>

def test_name():
    """Assert that the model's name is passed properly."""
    model = ATOMModel(HuberRegressor, acronym="hub")

    atom = ATOMRegressor(X_reg, y_reg)
    atom.run(model)
    assert hasattr(atom, "hub")


def test_fullname():
    """Assert that the model's fullname is passed properly."""
    model = ATOMModel(HuberRegressor, acronym="hub", fullname="Hubber")

    atom = ATOMRegressor(X_reg, y_reg)
    atom.run(model)
    assert atom.hub.fullname == "Hubber"


def test_needs_scaling():
    """Assert that the model's needs_scaling is passed properly."""
    model = ATOMModel(HuberRegressor, acronym="hub", needs_scaling=False)

    atom = ATOMRegressor(X_reg, y_reg)
    atom.run(model)
    assert not check_scaling(atom.hub.X)


def test_type():
    """Assert that the model's type is passed properly."""
    pytest.raises(ValueError, ATOMModel, HuberRegressor, type="test")
    model = ATOMModel(HuberRegressor, acronym="hub", type="linear")

    atom = ATOMRegressor(X_reg, y_reg)
    atom.run(model)
    assert atom.hub.type == "linear"


# Test ATOMLoader ================================================== >>

def test_invalid_verbose():
    """Assert that an error is raised when verbose is invalid."""
    pytest.raises(ValueError, ATOMLoader, FILE_DIR + "trainer", verbose=3)


def test_ATOMLoader():
    """Assert that the ATOMLoader function works as intended."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.save(FILE_DIR + "trainer")

    trainer2 = ATOMLoader(FILE_DIR + "trainer")
    assert trainer2.__class__.__name__ == "DirectClassifier"


def test_load_not_trainer():
    """Assert that an error is raised when data is provided without a trainer."""
    imputer = Imputer()
    imputer.save(FILE_DIR + "imputer")
    pytest.raises(TypeError, ATOMLoader, FILE_DIR + "imputer", data=(X_bin,))


def test_load_already_contains_data():
    """Assert that an error is raised when data is provided without needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + "atom", save_data=True)
    pytest.raises(ValueError, ATOMLoader, FILE_DIR + "atom", data=(X_bin,))


def test_data():
    """Assert that the method works when data is filled."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + "atom", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom", data=(X_bin, y_bin))
    assert atom2.dataset.equals(atom.dataset)


def test_n_rows():
    """Assert that n_rows is not used when transform_data=False."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.6, random_state=1)
    atom.save(FILE_DIR + "atom", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom", data=(X_bin, y_bin), transform_data=False)
    assert len(atom2.dataset) == len(X_bin)


def test_transform_data():
    """Assert that the data is transformed or not depending on the parameter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.outliers()
    atom.balance()
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
    atom.clean()
    atom.outliers()
    data_1 = atom.dataset
    atom.branch = "branch_2"
    atom.balance()
    atom.feature_generation(strategy="dfs", n_features=5)
    data_2 = atom.dataset
    atom.branch = "branch_3"
    atom.feature_selection(strategy="sfm", solver="lgb", n_features=10)
    data_3 = atom.dataset
    atom.save(FILE_DIR + "atom_2", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom_2", data=(X_bin, y_bin), transform_data=True)
    assert atom2._branches["main"].dataset.shape == data_1.shape
    assert atom2._branches["branch_2"].dataset.shape == data_2.shape
    assert atom2._branches["branch_3"].dataset.shape == data_3.shape


def test_multiple_branches():
    """Assert that the data is transformed over all branches."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "branch_2"
    atom.balance()
    atom.save(FILE_DIR + "atom", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom", data=(X_bin, y_bin), transform_data=True)
    assert len(atom2.branch.data) != len(X_bin)
    assert len(atom2._branches["main"].data) == len(X_bin)


def test_verbose_is_reset():
    """Assert that the verbosity of the estimator is reset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers()
    atom.save(FILE_DIR + "atom", save_data=False)

    atom2 = ATOMLoader(FILE_DIR + "atom", data=(X_bin, y_bin), verbose=2)
    assert atom2.pipeline[0].get_params()["verbose"] == 0


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
