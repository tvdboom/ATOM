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
from tensorflow.keras.datasets import mnist

# Own modules
from atom import ATOMClassifier
from atom.training import DirectClassifier
from atom.basetransformer import BaseTransformer
from atom.utils import merge
from .utils import (
    FILE_DIR, X_bin, y_bin, X_bin_array, y_bin_array, X10, y10, bin_train, bin_test
)


# Test properties ================================================== >>

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


@pytest.mark.parametrize("verbose", [-2, 3])
def test_verbose_parameter(verbose):
    """Assert that the verbose parameter is in correct range."""
    pytest.raises(ValueError, BaseTransformer, verbose=verbose)


def test_warnings_parameter_bool():
    """Assert that the warnings parameter works for a boolean value."""
    base = BaseTransformer(warnings=True)
    assert base.warnings == "default"

    base = BaseTransformer(warnings=False)
    assert base.warnings == "ignore"


def test_warnings_parameter_invalid_str():
    """Assert that an error is raised for an invalid string for warnings."""
    pytest.raises(ValueError, BaseTransformer, warnings="test")


def test_warnings_parameter_str():
    """Assert that the warnings parameter works for a boolean value."""
    base = BaseTransformer(warnings="always")
    assert base.warnings == "always"


def test_log_is_none():
    """Assert that no logging file is created when log=None."""
    BaseTransformer(logger=None)
    assert not glob.glob("log.log")


def test_create_log_file():
    """Assert that a logging file is created when log is not None."""
    BaseTransformer(logger=FILE_DIR + "log.log")
    assert glob.glob(FILE_DIR + "log.log")


def test_log_file_ends_with_log():
    """Assert that the logging file always ends with log."""
    BaseTransformer(logger=FILE_DIR + "logger")
    assert glob.glob(FILE_DIR + "logger.log")


def test_log_file_named_auto():
    """Assert that when log="auto", an automatic logging file is created."""
    BaseTransformer(logger=FILE_DIR + "auto")
    assert glob.glob(FILE_DIR + "BaseTransformer_*")


def test_logger_invalid_class():
    """Assert that an error is raised when logger is of invalid class."""
    pytest.raises(TypeError, BaseTransformer, logger=BaseTransformer)


def test_crash_with_logger():
    """Assert that the crash decorator works with a logger."""
    atom = ATOMClassifier(X_bin, y_bin, logger=FILE_DIR + "logger")
    pytest.raises(ValueError, atom.run, ["LR", "LDA"], n_calls=-1)


def test_random_state_setter():
    """Assert that an error is raised for a negative random_state."""
    pytest.raises(ValueError, BaseTransformer, random_state=-1)


# Test _prepare_input ============================================== >>

def test_input_data_in_atom():
    """Assert that the data does not change once in an atom pipeline."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    X10[3][2] = 99  # Change an item of the original variable
    assert 99 not in atom.dataset  # Is unchanged in the pipeline


def test_input_data_in_training():
    """Assert that the data does not change once in a training pipeline."""
    train = bin_train.copy()
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(train, bin_test)
    train.iloc[3, 2] = 99  # Change an item of the original variable
    assert 99 not in trainer.dataset  # Is unchanged in the pipeline


def test_multidimensional_X():
    """Assert that more than two dimensional datasets are handled correctly."""
    train, test = mnist.load_data()
    atom = ATOMClassifier(train, test, random_state=1)
    assert atom.X.columns == ['Features']
    assert atom.X.iloc[0, 0].shape == (28, 28)


def test_to_pandas():
    """Assert that the data provided is converted to pandas objects."""
    X, y = BaseTransformer._prepare_input(X_bin_array, y_bin_array)
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)


def test_y_is1dimensional():
    """Assert that an error is raised when y is not 1-dimensional."""
    y = [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]
    with pytest.raises(ValueError, match=r".*should be one-dimensional.*"):
        BaseTransformer._prepare_input(X10[:5], y)


def test_equal_length():
    """Assert that an error is raised when X and y don't have equal size."""
    with pytest.raises(ValueError, match=r".*number of rows.*"):
        BaseTransformer._prepare_input(X10, [0, 1, 1])


def test_equal_index():
    """Assert that an error is raised when X and y don't have same indices."""
    y = pd.Series(y_bin_array, index=range(10, len(y_bin_array) + 10))
    with pytest.raises(ValueError, match=r".*same indices.*"):
        BaseTransformer._prepare_input(X_bin, y)


def test_target_is_string():
    """Assert that the target column is assigned correctly for a string."""
    _, y = BaseTransformer._prepare_input(X_bin, y="mean radius")
    assert y.name == "mean radius"


def test_target_not_in_dataset():
    """Assert that the target column given by y is in X."""
    with pytest.raises(ValueError, match=r".*not found in X.*"):
        BaseTransformer._prepare_input(X_bin, "X")


def test_target_is_int():
    """Assert that target column is assigned correctly for an integer."""
    _, y = BaseTransformer._prepare_input(X_bin, y=0)
    assert y.name == "mean radius"


def test_target_is_none():
    """Assert that target column stays None when empty input."""
    _, y = BaseTransformer._prepare_input(X_bin, y=None)
    assert y is None


# Test _get_data_and_idx =========================================== >>

def test_empty_data_arrays():
    """Assert that an error is raised when no data is provided."""
    with pytest.raises(ValueError, match=r".*data arrays are empty.*"):
        ATOMClassifier(n_rows=100, random_state=1)


def test_input_is_X():
    """Assert that input X works as intended."""
    atom = ATOMClassifier(X_bin, random_state=1)
    assert atom.dataset.shape == X_bin.shape


def test_input_is_X_y():
    """Assert that input X, y works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dataset.shape == merge(X_bin, y_bin).shape


@pytest.mark.parametrize("n_rows", [0.7, 0.8, 1])
def test_n_rows_X_y_frac(n_rows):
    """Assert that n_rows<=1 work for input X and X, y."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=n_rows, random_state=1)
    assert len(atom.dataset) == int(len(X_bin) * n_rows)


def test_n_rows_X_y_int():
    """Assert that n_rows>1 work for input X and X, y."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=200, random_state=1)
    assert len(atom.dataset) == 200


@pytest.mark.parametrize("ts", [-2, 0, 1000])
def test_test_size_parameter(ts):
    """Assert that the test_size parameter is in correct range."""
    pytest.raises(ValueError, ATOMClassifier, X_bin, test_size=ts, random_state=1)


def test_test_size_fraction():
    """Assert that the test_size parameters splits the sets correctly when <1."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.2, random_state=1)
    assert len(atom.test) == int(0.2 * len(X_bin))
    assert len(atom.train) == len(X_bin) - int(0.2 * len(X_bin))


def test_test_size_int():
    """Assert that the test_size parameters splits the sets correctly when >=1."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=100, random_state=1)
    assert len(atom.test) == 100
    assert len(atom.train) == len(X_bin) - 100


def test_train_test_provided():
    """Assert that it runs when train and test are provided."""
    dataset = pd.concat([bin_train, bin_test]).reset_index(drop=True)

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.dataset.equals(dataset)


def test_train_test_as_tuple_provided():
    """Assert that it runs when (X_train, y_train), (X_test, y_test) is provided."""
    dataset = pd.concat([bin_train, bin_test]).reset_index(drop=True)
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run((X_train, y_train), (X_test, y_test))
    assert trainer.dataset.equals(dataset)


def test_4_data_provided():
    """Assert that it runs when X_train, X_test, etc... are provided."""
    dataset = pd.concat([bin_train, bin_test]).reset_index(drop=True)
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(X_train, X_test, y_train, y_test)
    assert trainer.dataset.equals(dataset)


def test_invalid_input():
    """Assert that an error is raised when input arrays are invalid."""
    trainer = DirectClassifier("LR", random_state=1)
    pytest.raises(ValueError, trainer.run, X_bin, bin_train, bin_test)


def test_n_rows_train_test_frac():
    """Assert that n_rows<=1 work for input with train and test."""
    atom = ATOMClassifier(bin_train, bin_test, n_rows=0.8, random_state=1)
    assert len(atom.train) == round(len(bin_train) * 0.8)
    assert len(atom.test) == round(len(bin_test) * 0.8)


def test_n_rows_train_test_int():
    """Assert that an error is raised when n_rows>1 for input with train and test."""
    with pytest.raises(ValueError, match=r".*should be <=1 when train and test.*"):
        ATOMClassifier(bin_train, bin_test, n_rows=100, random_state=1)


def test_dataset_is_shuffled():
    """Assert that the dataset is shuffled before splitting."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert not X_bin.equals(atom.X)


def test_reset_index():
    """Assert that the indices are reset for the whole dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert list(atom.dataset.index) == list(range(len(X_bin)))


# Test log ========================================================= >>

def test_log():
    """Assert the log method works."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=2, logger=FILE_DIR + "auto")
    atom.log("test", 1)


# Test save ======================================================== >>

def test_file_is_saved():
    """Assert that the pickle file is created."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.save(FILE_DIR + "auto")
    assert glob.glob(FILE_DIR + "ATOMClassifier")


def test_save_data():
    """Assert that the pickle file is created for save_data=False."""
    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin)
    dataset = atom.dataset.copy()
    atom.save(filename=FILE_DIR + "atom", save_data=False)
    assert atom.dataset.equals(dataset)

    # From a trainer
    trainer = DirectClassifier("LR")
    trainer.run(bin_train, bin_test)
    dataset = trainer.dataset.copy()
    trainer.save(filename=FILE_DIR + "trainer", save_data=False)
    assert trainer.dataset.equals(dataset)
