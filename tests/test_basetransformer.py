# coding: utf-8

"""Automated Tool for Optimized Modelling (ATOM)

Author: tvdboom
Description: Unit tests for basetransformer.py

"""

# Standard packages
import glob
import pytest
import pandas as pd
import multiprocessing

# Own modules
from atom import ATOMClassifier
from atom.training import TrainerClassifier
from atom.basetransformer import BaseTransformer
from .utils import (
    FILE_DIR, X_bin, y_bin, X_bin_array, y_bin_array, X10, bin_train, bin_test
    )


# Test properties =========================================================== >>

def test_n_jobs_maximum_cores():
    """Assert that value equals n_cores if maximum is exceeded."""
    base = BaseTransformer(n_jobs=1000)
    assert base.n_jobs == multiprocessing.cpu_count()


def test_n_jobs_is_zero():
    """Assert that an error is raised when n_jobs=0."""
    pytest.raises(ValueError, BaseTransformer, n_jobs=0)


def test_too_far_negative_n_jobs():
    """Assert that an error is raised when value too far negative."""
    pytest.raises(ValueError, BaseTransformer, n_jobs=-1000)


def test_negative_n_jobs():
    """Assert that value is set correctly for negative values."""
    base = BaseTransformer(n_jobs=-1)
    assert base.n_jobs == multiprocessing.cpu_count()

    base = BaseTransformer(n_jobs=-2)
    assert base.n_jobs == multiprocessing.cpu_count() - 1


@pytest.mark.parametrize('verbose', [-2, 3])
def test_verbose_parameter(verbose):
    """Assert that the verbose parameter is in correct range."""
    pytest.raises(ValueError, BaseTransformer, verbose=verbose)


def test_warnings_parameter_bool():
    """Assert that the warnings parameter works for a boolean value."""
    base = BaseTransformer(warnings=True)
    assert base.warnings == 'default'

    base = BaseTransformer(warnings=False)
    assert base.warnings == 'ignore'


def test_warnings_parameter_invalid_str():
    """Assert that an error is raised for an invalid string for warnings."""
    pytest.raises(ValueError, BaseTransformer, warnings='test')


def test_warnings_parameter_str():
    """Assert that the warnings parameter works for a boolean value."""
    base = BaseTransformer(warnings='always')
    assert base.warnings == 'always'


def test_log_is_none():
    """Assert that no logging file is created when log=None."""
    BaseTransformer(logger=None)
    assert not glob.glob('log.log')


def test_create_log_file():
    """Assert that a logging file is created when log is not None."""
    BaseTransformer(logger=FILE_DIR + 'log.log')
    assert glob.glob(FILE_DIR + 'log.log')


def test_log_file_ends_with_log():
    """Assert that the logging file always ends with log."""
    BaseTransformer(logger=FILE_DIR + 'logger')
    assert glob.glob(FILE_DIR + 'logger.log')


def test_log_file_named_auto():
    """Assert that when log='auto', an automatic logging file is created."""
    BaseTransformer(logger=FILE_DIR + 'auto')
    assert glob.glob(FILE_DIR + 'BaseTransformer_logger_*')


def test_logger_invalid_class():
    """Assert that an error is raised when logger is of invalid class."""
    pytest.raises(TypeError, BaseTransformer, logger=BaseTransformer)


def test_crash_with_logger():
    """Assert that the crash decorator works with a logger."""
    atom = ATOMClassifier(X_bin, y_bin, logger=FILE_DIR + 'logger')
    pytest.raises(ValueError, atom.run, ['LR', 'LDA'], n_calls=-1)


def test_random_state_setter():
    """Assert that an error is raised for a negative random_state."""
    pytest.raises(ValueError, BaseTransformer, random_state=-1)


# Test _prepare_input ======================================================= >>

def test_copy_input_data():
    """Assert that the prepare_input method uses copies of the data."""
    X, y = BaseTransformer()._prepare_input(X_bin, y_bin)
    assert X is not X_bin and y is not y_bin


def test_to_pandas():
    """Assert that the data provided is converted to pandas objects."""
    X, y = BaseTransformer._prepare_input(X_bin_array, y_bin_array)
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)


def test_equal_length():
    """Assert that an error is raised when X and y don't have equal size."""
    pytest.raises(ValueError, BaseTransformer._prepare_input, X10, [0, 1, 1])


def test_y_is1dimensional():
    """Assert that an error is raised when y is not 1-dimensional."""
    y = [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0],
         [1, 1], [1, 0], [0, 1], [1, 1], [1, 0]]
    pytest.raises(ValueError, BaseTransformer._prepare_input, X10, y)


def test_equal_index():
    """Assert that an error is raised when X and y don't have same indices."""
    y = pd.Series(y_bin_array, index=range(10, len(y_bin_array)+10))
    pytest.raises(ValueError, BaseTransformer._prepare_input, X_bin, y)


def test_target_is_string():
    """Assert that the target column is assigned correctly for a string."""
    _, y = BaseTransformer._prepare_input(X_bin, y='mean radius')
    assert y.name == 'mean radius'


def test_target_not_in_dataset():
    """Assert that the target column given by y is in X."""
    pytest.raises(ValueError, BaseTransformer._prepare_input, X_bin, 'X')


def test_target_is_int():
    """Assert that target column is assigned correctly for an integer."""
    _, y = BaseTransformer._prepare_input(X_bin, y=0)
    assert y.name == 'mean radius'


def test_target_is_none():
    """Assert that target column stays None when empty input."""
    _, y = BaseTransformer._prepare_input(X_bin, y=None)
    assert y is None


# Test log ================================================================== >>

def test_log():
    """Assert the log method works."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=2, logger=FILE_DIR + 'auto')
    atom.log('test', 1)


# Test save ================================================================= >>

def test_file_is_saved():
    """Assert that the pickle file is created."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.save(FILE_DIR + 'auto')
    assert glob.glob(FILE_DIR + 'ATOMClassifier')


def test_save_data():
    """Assert that the pickle file is created for save_data=False."""
    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin)
    dataset = atom.dataset.copy()
    atom.save(filename=FILE_DIR + 'atom', save_data=False)
    assert atom.dataset.equals(dataset)

    # From a trainer
    trainer = TrainerClassifier('LR')
    trainer.run(bin_train, bin_test)
    dataset = trainer.dataset.copy()
    trainer.save(filename=FILE_DIR + 'trainer', save_data=False)
    assert trainer.dataset.equals(dataset)
