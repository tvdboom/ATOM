# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the __init__ method of the ATOM class.

"""

# Import packages
import glob
import pytest
import numpy as np
import multiprocessing
from atom import ATOMClassifier, ATOMRegressor
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg


# << ======================= Tests ========================= >>

def test_log_is_none():
    """Assert that no logging file is created when log=None."""
    ATOMClassifier(X_bin, y_bin, log=None)
    assert not glob.glob('log.log')


def test_create_log_file():
    """Assert that a logging file is created when log is not None."""
    ATOMClassifier(X_bin, y_bin, log='logger.log')
    assert glob.glob('logger.log')


def test_log_file_ends_with_log():
    """Assert that the logging file always ends with log."""
    ATOMClassifier(X_bin, y_bin, log='logger')
    assert glob.glob('logger.log')


def test_log_file_named_auto():
    """Assert that when log='auto', an automatic logging file is created."""
    ATOMClassifier(X_bin, y_bin, log='auto')
    assert glob.glob('ATOM_logger_*')


def test_goal_assigning():
    """Assert that the goal attribute is assigned correctly."""
    atom = ATOMClassifier(X_bin, y_bin)
    assert atom.goal == 'classification'

    atom = ATOMRegressor(X_reg, y_reg)
    assert atom.goal == 'regression'


def test_percentage_parameter():
    """Assert that the percentage parameter is in correct range."""
    for p in [0, -1, 120]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, percentage=p)


def test_test_size_parameter():
    """Assert that the test_size parameter is in correct range."""
    for size in [0., -3.1, 12.2]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, test_size=size)


def test_verbose_parameter():
    """Assert that the verbose parameter is in correct range."""
    for vb in [-2, 4]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, verbose=vb)


def test_n_jobs_maximum_cores():
    """Assert that n_jobs equals n_cores if maximum is exceeded."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=1000)
    assert atom.n_jobs == multiprocessing.cpu_count()


def test_n_jobs_is_zero():
    """Assert that when n_jobs=0, 1 core is used."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=0)
    assert atom.n_jobs == 1


def test_too_far_negative_n_jobs():
    """Assert that an error is raised when n_jobs too far negative."""
    pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, n_jobs=-1000)


def test_negative_n_jobs():
    """Assert that n_jobs is set correctly for negative values."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1)
    assert atom.n_jobs == multiprocessing.cpu_count()

    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-3)
    assert atom.n_jobs == multiprocessing.cpu_count() - 2


def test_random_state_parameter():
    """Assert the return of same results for two independent runs."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom.pipeline(['lr', 'lgb', 'pa'], 'f1', n_calls=8)
    atom2 = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom2.pipeline(['lr', 'lgb', 'pa'], 'f1', n_calls=8)

    assert atom.lr.score_test == atom2.lr.score_test
    assert atom.lgb.score_test == atom2.lgb.score_test
    assert atom.pa.score_test == atom2.pa.score_test


def test_task_assigning():
    """Assert that the task attribute is assigned correctly."""
    atom = ATOMClassifier(X_bin, y_bin)
    assert atom.task == 'binary classification'

    atom = ATOMClassifier(X_class, y_class)
    assert atom.task == 'multiclass classification'

    atom = ATOMRegressor(X_reg, y_reg)
    assert atom.task == 'regression'


def test_merger_to_dataset():
    """Assert that the merger between X and y was successful."""
    atom = ATOMClassifier(X_bin, y_bin)
    merger = X_bin.merge(
        y_bin.astype(np.int64).to_frame(), left_index=True, right_index=True
        )

    # Order of rows can be different
    df1 = merger.sort_values(by=merger.columns.tolist())
    df1.reset_index(drop=True, inplace=True)
    df2 = atom.dataset.sort_values(by=atom.dataset.columns.tolist())
    df2.reset_index(drop=True, inplace=True)
    assert df1.equals(df2)


def test_target_is_last_column():
    """Assert that target is placed as the last column of the dataset."""
    atom = ATOMClassifier(X_bin, 'mean radius')
    assert atom.dataset.columns[-1] == 'mean radius'
