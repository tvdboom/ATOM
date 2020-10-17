# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for api.py

"""

# Import packages
import pytest

# Own modules
from atom import ATOMClassifier, ATOMRegressor, ATOMLoader
from atom.training import TrainerClassifier
from atom.data_cleaning import Imputer
from atom.utils import merge
from .utils import FILE_DIR, X_bin, y_bin, X_reg, y_reg


# Test ATOMLoader =========================================================== >>

def test_invalid_verbose():
    """Assert that an error is raised when verbose is invalid."""
    pytest.raises(ValueError, ATOMLoader, FILE_DIR + 'trainer', verbose=3)


def test_ATOMLoader():
    """Assert that the ATOMLoader function works as intended."""
    trainer = TrainerClassifier('LR', random_state=1)
    trainer.save(FILE_DIR + 'trainer')

    trainer2 = ATOMLoader(FILE_DIR + 'trainer')
    assert trainer2.__class__.__name__ == 'TrainerClassifier'


def test_new_data():
    """Assert that the ATOMLoader function works when new data is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + 'atom', save_data=False)

    atom2 = ATOMLoader(FILE_DIR + 'atom', merge(X_bin, y_bin))
    assert atom2.dataset.equals(atom.dataset)


def test_load_already_contains_data():
    """Assert that an error is raised when data is provided without needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + 'atom', save_data=True)
    pytest.raises(ValueError, ATOMLoader, FILE_DIR + 'atom', X_bin)


def test_load_not_atom():
    """Assert that an error is raised when data is provided without _data attr."""
    imputer = Imputer()
    imputer.save(FILE_DIR + 'imputer')
    pytest.raises(TypeError, ATOMLoader, FILE_DIR + 'imputer', X_bin)


def test_data_is_X_y():
    """Assert that the method works when data contains y."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + 'atom', save_data=False)

    atom2 = ATOMLoader(FILE_DIR + 'atom', X_bin, y_bin)
    assert atom2.dataset.equals(atom.dataset)


def test_n_rows():
    """Assert that the method when n_rows is provided."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.6, random_state=1)
    atom.save(FILE_DIR + 'atom', save_data=False)

    atom2 = ATOMLoader(FILE_DIR + 'atom', X_bin, y_bin, n_rows=0.6)
    assert atom2.dataset.equals(atom.dataset)


def test_transform_data():
    """Assert that the data is transformed or not depending on the parameter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers()
    atom.balance()
    atom.feature_generation(strategy='dfs', n_features=5)
    atom.feature_selection(strategy='sfm', solver='lgb', n_features=10)
    atom.save(FILE_DIR + 'atom', save_data=False)

    atom2 = ATOMLoader(FILE_DIR + 'atom', X_bin, y_bin, transform_data=True)
    assert atom2.shape[0] != X_bin.shape[0] and atom2.shape[1] != X_bin.shape[1] + 1

    atom3 = ATOMLoader(FILE_DIR + 'atom', X_bin, y_bin, transform_data=False)
    assert atom3.shape[0] == X_bin.shape[0] and atom3.shape[1] == X_bin.shape[1] + 1


def test_verbose_is_reset():
    """Assert that the verbosity of the estimator is reset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers()
    atom.save(FILE_DIR + 'atom', save_data=False)

    atom2 = ATOMLoader(FILE_DIR + 'atom', X_bin, y_bin, verbose=2)
    assert atom2.pipeline[1].get_params()['verbose'] == 0


def test_trainer_gets_data():
    """Assert that the trainer gets ATOM'S data."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR')
    atom.save(FILE_DIR + 'atom', save_data=False)

    atom2 = ATOMLoader(FILE_DIR + 'atom', X_bin, y_bin)
    assert atom2.trainer._data is atom2._data


# Test ATOMClassifier ======================================================= >>

def test_goal_ATOMClassifier():
    """Assert that the goal is set correctly for ATOMClassifier."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.goal == 'classification'


# Test ATOMRegressor ======================================================== >>

def test_goal_ATOMRegressor():
    """Assert that the goal is set correctly for ATOMRegressor."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.goal == 'regression'
