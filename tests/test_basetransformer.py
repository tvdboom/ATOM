# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM)

Author: Mavs
Description: Unit tests for basetransformer.py

"""

import glob
import multiprocessing
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sklearnex
from pandas.testing import assert_frame_equal
from sklearn.naive_bayes import GaussianNB
from sklearnex.svm import SVC

from atom import ATOMClassifier, ATOMRegressor
from atom.basetransformer import BaseTransformer
from atom.training import DirectClassifier
from atom.utils import merge

from .conftest import (
    X10, X_bin, X_bin_array, X_idx, X_sparse, X_text, bin_test, bin_train, y10,
    y_bin, y_bin_array, y_idx,
)


# Test properties ================================================== >>

def test_n_jobs_maximum_cores():
    """Assert that value equals n_cores if maximum is exceeded."""
    base = BaseTransformer(n_jobs=1000)
    assert base.n_jobs == multiprocessing.cpu_count()


@pytest.mark.parametrize("n_jobs", [0, -1000])
def test_n_jobs_invalid(n_jobs):
    """Assert that an error is raised when n_jobs is invalid."""
    with pytest.raises(ValueError, match=".*n_jobs parameter.*"):
        BaseTransformer(n_jobs=n_jobs)


def test_negative_n_jobs():
    """Assert that value is set correctly for negative values."""
    base = BaseTransformer(n_jobs=-1)
    assert base.n_jobs == multiprocessing.cpu_count()

    base = BaseTransformer(n_jobs=-2)
    assert base.n_jobs == multiprocessing.cpu_count() - 1


def test_device_parameter():
    """Assert that the device is set correctly."""
    BaseTransformer(device="gpu")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_engine_parameter_sklearnex():
    """Assert that sklearnex offloads to the right device."""
    BaseTransformer(device="gpu", engine="sklearnex")
    assert sklearnex.get_config()["target_offload"] == "gpu"


def test_engine_parameter_cuml_with_cpu():
    """Assert that an error is raised when device='cpu' and engine='cuml'."""
    with pytest.raises(ValueError, match=".*only supports sklearn.*"):
        BaseTransformer(device="cpu", engine="cuml")


def test_engine_parameter_no_cuml():
    """Assert that an error is raised when cuml is not installed."""
    with pytest.raises(ModuleNotFoundError, match=".*Failed to import cuml.*"):
        BaseTransformer(device="gpu", engine="cuml")


def test_engine_parameter_invalid():
    """Assert that an error is raised when engine is invalid."""
    with pytest.raises(ValueError, match=".*Choose from : sklearn.*"):
        BaseTransformer(engine="invalid")


@pytest.mark.parametrize("verbose", [-2, 3])
def test_verbose_parameter(verbose):
    """Assert that the verbose parameter is in correct range."""
    with pytest.raises(ValueError, match=".*verbose parameter.*"):
        BaseTransformer(verbose=verbose)


def test_warnings_parameter_bool():
    """Assert that the warnings parameter works for a bool."""
    base = BaseTransformer(warnings=True)
    assert base.warnings == "default"

    base = BaseTransformer(warnings=False)
    assert base.warnings == "ignore"


def test_warnings_parameter_invalid_str():
    """Assert that an error is raised for an invalid string for warnings."""
    with pytest.raises(ValueError, match=".*warnings parameter.*"):
        BaseTransformer(warnings="test")


def test_warnings_parameter_str():
    """Assert that the warnings parameter works for a str."""
    base = BaseTransformer(warnings="always")
    assert base.warnings == "always"


@patch("atom.utils.logging.getLogger")
def test_logger_creator(cls):
    """Assert that the logger is created correctly."""
    BaseTransformer(logger=None)
    cls.assert_not_called()

    BaseTransformer(logger="auto")
    cls.assert_called_once()


@patch("atom.utils.logging.getLogger")
def test_crash_with_logger(cls):
    """Assert that the crash decorator works with a logger."""
    atom = ATOMClassifier(X_bin, y_bin, logger="log")
    pytest.raises(ValueError, atom.run, ["LR", "LDA"], n_calls=-1)
    cls.return_value.exception.assert_called()


@patch("mlflow.set_experiment")
def test_experiment_creation(mlflow):
    """Assert that the mlflow experiment is created."""
    base = BaseTransformer(experiment="test")
    assert base.experiment == "test"
    mlflow.assert_called_once()


def test_random_state_setter():
    """Assert that an error is raised for a negative random_state."""
    with pytest.raises(ValueError, match=".*random_state parameter.*"):
        BaseTransformer(random_state=-1)


def test_device_id_no_value():
    """Assert that the device id can be left empty."""
    base = BaseTransformer(device="gpu")
    assert base._device_id == 0


def test_device_id_int():
    """Assert that the device id can be set."""
    base = BaseTransformer(device="gpu:2")
    assert base._device_id == 2


def test_device_id_invalid():
    """Assert that an error is raised when the device id is invalid."""
    with pytest.raises(ValueError, match=".*Use a single integer.*"):
        BaseTransformer(device="gpu:2,3")


# Test _get_est_class ============================================== >>

def test_get_est_class_from_engine():
    """Assert that the class can be retrieved from an engine."""
    base = BaseTransformer(device="cpu", engine="sklearnex")
    assert base._get_est_class("SVC", "svm") == SVC


def test_get_est_class_from_default():
    """Assert that the class is retrieved from sklearn when import fails."""
    base = BaseTransformer(device="cpu", engine="sklearnex")
    assert base._get_est_class("GaussianNB", "naive_bayes") == GaussianNB


# Test _prepare_input ============================================== >>

def test_input_X_and_y_None():
    """Assert that an error is raised when both X and y are None."""
    with pytest.raises(ValueError, match=".*both None.*"):
        BaseTransformer._prepare_input()


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
    train.iat[3, 2] = 99  # Change an item of the original variable
    assert 99 not in trainer.dataset  # Is unchanged in the pipeline


def test_text_to_corpus():
    """Assert that for text data the column is named corpus."""
    atom = ATOMClassifier(X_text, y10, random_state=1)
    assert atom.X.columns == ["corpus"]


def test_int_columns_to_str():
    """Assert that int columns are converted to str."""
    X = X_bin.copy()
    X.columns = range(X.shape[1])
    atom = ATOMClassifier(X, y_bin, random_state=1)
    assert atom.X.columns[0] == "0"


def test_sparse_matrices_X_y():
    """Assert that sparse matrices are accepted as (X, y) input."""
    atom = ATOMClassifier(X_sparse, y10, random_state=1)
    assert isinstance(atom.X, pd.DataFrame)
    assert atom.shape == (10, 4)
    assert atom[atom.columns[0]].dtype.name == "Sparse[int64, 0]"


def test_sparse_matrices_2_tuples():
    """Assert that sparse matrices are accepted as 2-tuples input."""
    atom = ATOMClassifier((X_sparse, y10), (X_sparse, y10), random_state=1)
    assert isinstance(atom.X, pd.DataFrame)
    assert atom.shape == (20, 4)
    assert atom[atom.columns[0]].dtype.name == "Sparse[int64, 0]"


def test_to_pandas():
    """Assert that the data provided is converted to pandas objects."""
    X, y = BaseTransformer._prepare_input(X_bin_array, y_bin_array)
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)


def test_y_is1dimensional():
    """Assert that an error is raised when y is not 1-dimensional."""
    y = [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]
    with pytest.raises(ValueError, match=".*should be one-dimensional.*"):
        BaseTransformer._prepare_input(X10[:5], y)


def test_equal_length():
    """Assert that an error is raised when X and y don't have equal size."""
    with pytest.raises(ValueError, match=".*number of rows.*"):
        BaseTransformer._prepare_input(X10, [0, 1, 1])


def test_equal_index():
    """Assert that an error is raised when X and y don't have same indices."""
    y = pd.Series(y_bin_array, index=range(10, len(y_bin_array) + 10))
    with pytest.raises(ValueError, match=".*same indices.*"):
        BaseTransformer._prepare_input(X_bin, y)


def test_target_is_string():
    """Assert that the target column is assigned correctly for a string."""
    _, y = BaseTransformer._prepare_input(X_bin, y="mean radius")
    assert y.name == "mean radius"


def test_target_not_in_dataset():
    """Assert that the target column given by y is in X."""
    with pytest.raises(ValueError, match=".*not found in X.*"):
        BaseTransformer._prepare_input(X_bin, "X")


def test_X_is_None_with_str():
    """Assert that an error is raised when X is None and y is a string."""
    with pytest.raises(ValueError, match=".*can't be None when y is a str.*"):
        BaseTransformer._prepare_input(y="test")


def test_target_is_int():
    """Assert that target column is assigned correctly for an integer."""
    _, y = BaseTransformer._prepare_input(X_bin, y=0)
    assert y.name == "mean radius"


def test_X_is_None_with_int():
    """Assert that an error is raised when X is None and y is an int."""
    with pytest.raises(ValueError, match=".*can't be None when y is an int.*"):
        BaseTransformer._prepare_input(y=1)


def test_target_is_none():
    """Assert that target column stays None when empty input."""
    _, y = BaseTransformer._prepare_input(X_bin, y=None)
    assert y is None


# Test _set_index ================================================== >>

def test_index_is_true():
    """Assert that the indices are left as is when index=True."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, shuffle=False, random_state=1)
    assert atom.dataset.index[0] == "index_0"


def test_index_is_False():
    """Assert that the indices are reset when index=False."""
    atom = ATOMClassifier(X_idx, y_idx, index=False, shuffle=False, random_state=1)
    assert atom.dataset.index[0] == 0


def test_index_is_int_invalid():
    """Assert that an error is raised when the index is an invalid int."""
    with pytest.raises(ValueError, match=".*is out of range.*"):
        ATOMClassifier(X_bin, y_bin, index=1000, random_state=1)


def test_index_is_int():
    """Assert that a column can be selected from a position."""
    atom = ATOMClassifier(X_bin, y_bin, index=0, random_state=1)
    assert atom.dataset.index.name == "mean radius"


def test_index_is_str_invalid():
    """Assert that an error is raised when the index is an invalid str."""
    with pytest.raises(ValueError, match=".*not found in the dataset.*"):
        ATOMClassifier(X_bin, y_bin, index="invalid", random_state=1)


def test_index_is_str():
    """Assert that a column can be selected from a name."""
    atom = ATOMClassifier(X_bin, y_bin, index="mean texture", random_state=1)
    assert atom.dataset.index.name == "mean texture"


def test_index_is_target():
    """Assert that an error is raised when the index is the target column."""
    with pytest.raises(ValueError, match=".*same as the target column.*"):
        ATOMRegressor(X_bin, index="worst fractal dimension", random_state=1)


def test_index_is_sequence_no_data_sets_invalid_length():
    """Assert that an error is raised when len(index) != len(data)."""
    with pytest.raises(ValueError, match=".*Length of index.*"):
        ATOMClassifier(X_bin, y_bin, index=[1, 2, 3], random_state=1)


def test_index_is_sequence_no_data_sets():
    """Assert that a sequence is set as index when provided."""
    index = [f"index_{i}" for i in range(len(X_bin))]
    atom = ATOMClassifier(X_bin, y_bin, index=index, random_state=1)
    assert atom.dataset.index[0] == "index_190"


def test_index_is_sequence_has_data_sets_invalid_length():
    """Assert that an error is raised when len(index) != len(data)."""
    with pytest.raises(ValueError, match=".*Length of index.*"):
        ATOMClassifier(bin_train, bin_test, index=[1, 2, 3], random_state=1)


def test_index_is_sequence_has_data_sets():
    """Assert that a sequence is set as index when provided."""
    index = [f"index_{i}" for i in range(len(bin_train) + 2 * len(bin_test))]
    atom = ATOMClassifier(bin_train, bin_test, bin_test, index=index, random_state=1)
    assert atom.dataset.index[0] == "index_174"
    assert atom.holdout.index[0] == "index_661"


# Test _get_stratify_columns======================================== >>

@pytest.mark.parametrize("stratify", [False, True, -1, "target", [-1]])
def test_stratify_options(stratify):
    """Assert that the data can be stratified among data sets."""
    atom = ATOMClassifier(X_bin, y_bin, stratify=stratify, random_state=1)
    train_balance = atom.classes["train"][0] / atom.classes["train"][1]
    test_balance = atom.classes["test"][0] / atom.classes["test"][1]
    np.testing.assert_almost_equal(train_balance, test_balance, decimal=2)


def test_stratify_invalid_column_int():
    """Assert that an error is raised when the value is invalid."""
    with pytest.raises(ValueError, match=".*out of range for a dataset.*"):
        ATOMClassifier(X_bin, y_bin, stratify=100, random_state=1)


def test_stratify_invalid_column_str():
    """Assert that an error is raised when the value is invalid."""
    with pytest.raises(ValueError, match=".*not found in the dataset.*"):
        ATOMClassifier(X_bin, y_bin, stratify="invalid", random_state=1)


# Test _get_data =================================================== >>

def test_empty_data_arrays():
    """Assert that an error is raised when no data is provided."""
    with pytest.raises(ValueError, match=".*data arrays are empty.*"):
        ATOMClassifier(n_rows=100, random_state=1)


def test_data_already_set():
    """Assert that if there already is data, the call to run can be empty."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    trainer.run()
    pd.testing.assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))
    pd.testing.assert_index_equal(trainer.branch._idx[0], bin_train.index)
    pd.testing.assert_index_equal(trainer.branch._idx[1], bin_test.index)


def test_input_is_X():
    """Assert that input X works."""
    atom = ATOMRegressor(X_bin, random_state=1)
    assert atom.dataset.shape == X_bin.shape


def test_input_is_X_with_parameter_y():
    """Assert that input X can be combined with parameter y."""
    atom = ATOMRegressor(X_bin, y="mean texture", random_state=1)
    assert atom.target == "mean texture"


def test_input_invalid_holdout():
    """Assert that an error is raised when holdout is invalid."""
    with pytest.raises(ValueError, match=".*holdout_size parameter.*"):
        ATOMClassifier(X_bin, test_size=0.3, holdout_size=0.8)


@pytest.mark.parametrize("holdout_size", [0.1, 40])
def test_input_is_X_with_holdout(holdout_size):
    """Assert that input X can be combined with a holdout set."""
    atom = ATOMRegressor(X_bin, holdout_size=holdout_size, random_state=1)
    assert isinstance(atom.holdout, pd.DataFrame)


@pytest.mark.parametrize("shuffle", [True, False])
def test_input_is_train_test_with_holdout(shuffle):
    """Assert that input train and test can be combined with a holdout set."""
    atom = ATOMClassifier(bin_train, bin_test, bin_test, shuffle=shuffle)
    assert isinstance(atom.holdout, pd.DataFrame)


@pytest.mark.parametrize("n_rows", [0.7, 0.8, 1])
def test_n_rows_X_y_frac(n_rows):
    """Assert that n_rows<=1 work for input X and X, y."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=n_rows, random_state=1)
    assert len(atom.dataset) == int(len(X_bin) * n_rows)


def test_n_rows_X_y_int():
    """Assert that n_rows>1 work for input X and X, y."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=200, random_state=1)
    assert len(atom.dataset) == 200


def test_n_rows_too_large():
    """Assert that an error is raised when n_rows>len(data)."""
    with pytest.raises(ValueError, match=".*n_rows parameter.*"):
        ATOMClassifier(X_bin, y_bin, n_rows=1e6, random_state=1)


def test_no_shuffle_X_y():
    """Assert that the order is kept when shuffle=False."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, n_rows=30)
    assert_frame_equal(atom.X, X_bin.iloc[:30, :])


def test_length_dataset():
    """Assert that the dataset is always len>=2."""
    with pytest.raises(ValueError, match=".*n_rows parameter.*"):
        ATOMClassifier(X10, y10, n_rows=0.01, random_state=1)


@pytest.mark.parametrize("test_size", [-2, 0, 1000])
def test_test_size_parameter(test_size):
    """Assert that the test_size parameter is in correct range."""
    with pytest.raises(ValueError, match=".*test_size parameter.*"):
        ATOMClassifier(X_bin, test_size=test_size, random_state=1)


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


def test_input_is_X_y():
    """Assert that input X, y works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dataset.shape == merge(X_bin, y_bin).shape


def test_input_is_2_tuples():
    """Assert that the 2 tuples input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run((X_train, y_train), (X_test, y_test))
    pd.testing.assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_input_is_train_test():
    """Assert that input train, test works."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    pd.testing.assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_input_is_train_test_with_parameter_y():
    """Assert that input X works can be combined with y."""
    atom = ATOMClassifier(bin_train, bin_test, y="mean texture", random_state=1)
    assert atom.target == "mean texture"


def test_input_is_3_tuples():
    """Assert that the 3 tuples input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]
    X_holdout = bin_test.iloc[:, :-1]
    y_holdout = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run((X_train, y_train), (X_test, y_test), (X_holdout, y_holdout))
    pd.testing.assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))
    pd.testing.assert_frame_equal(trainer.holdout, bin_test)


def test_input_is_train_test_holdout():
    """Assert that input train, test, holdout works."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test, bin_test)
    pd.testing.assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))
    pd.testing.assert_frame_equal(trainer.holdout, bin_test)


def test_4_data_provided():
    """Assert that the 4 elements input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(X_train, X_test, y_train, y_test)
    pd.testing.assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_6_data_provided():
    """Assert that the 6 elements input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]
    X_holdout = bin_test.iloc[:, :-1]
    y_holdout = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(X_train, X_test, X_holdout, y_train, y_test, y_holdout)
    pd.testing.assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))
    pd.testing.assert_frame_equal(trainer.holdout, bin_test)


def test_invalid_input():
    """Assert that an error is raised when input arrays are invalid."""
    trainer = DirectClassifier("LR", random_state=1)
    with pytest.raises(ValueError, match=".*Invalid data arrays.*"):
        trainer.run(X_bin, y_bin, X_bin, y_bin, y_bin, X_bin, X_bin)


def test_n_rows_train_test_frac():
    """Assert that n_rows<=1 work for input with train and test."""
    atom = ATOMClassifier(bin_train, bin_test, n_rows=0.8, random_state=1)
    assert len(atom.train) == 318
    assert len(atom.test) == 136


def test_no_shuffle_train_test():
    """Assert that the order is kept when shuffle=False."""
    atom = ATOMClassifier(bin_train, bin_test, shuffle=False)
    assert_frame_equal(atom.train, bin_train.reset_index(drop=True), check_dtype=False)


def test_n_rows_train_test_int():
    """Assert that an error is raised when n_rows>1 for input with train and test."""
    with pytest.raises(ValueError, match=".*has to be <1 when a train and test.*"):
        ATOMClassifier(bin_train, bin_test, n_rows=100, random_state=1)


def test_dataset_is_shuffled():
    """Assert that the dataset is shuffled before splitting."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=True, random_state=1)
    assert not X_bin.equals(atom.X)


def test_holdout_is_shuffled():
    """Assert that the holdout set is shuffled."""
    atom = ATOMClassifier(bin_train, bin_test, bin_test, shuffle=True, random_state=1)
    assert not bin_test.equals(atom.holdout)


def test_reset_index():
    """Assert that the indices are reset for the all data sets."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    assert list(atom.dataset.index) == list(range(len(atom.dataset)))
    assert list(atom.holdout.index) == list(range(len(atom.holdout)))


def test_unequal_columns_train_test():
    """Assert that an error is raised when train and test have different columns."""
    with pytest.raises(ValueError, match=".*train and test set do not have.*"):
        ATOMClassifier(X10, bin_test, random_state=1)


def test_unequal_columns_holdout():
    """Assert that an error is raised when holdout has different columns."""
    with pytest.raises(ValueError, match=".*holdout set does not have.*"):
        ATOMClassifier(bin_train, bin_test, X10, random_state=1)


def test_merger_to_dataset():
    """Assert that the merger between X and y was successful."""
    # Reset index since order of rows is different after shuffling
    merger = X_bin.merge(y_bin.to_frame(), left_index=True, right_index=True)
    df1 = merger.sort_values(by=merger.columns.tolist())

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    df2 = atom.dataset.sort_values(by=atom.dataset.columns.tolist())
    assert_frame_equal(
        left=df1.reset_index(drop=True),
        right=df2.reset_index(drop=True),
        check_dtype=False,
    )


# Test log ========================================================= >>

def test_log_invalid_severity():
    """Assert that an error is raised when the severity is invalid."""
    with pytest.raises(ValueError, match=".*severity parameter.*"):
        BaseTransformer(logger="log").log("test", severity="invalid")


@patch("atom.utils.logging.getLogger")
def test_log(cls):
    """Assert the log method works."""
    base = BaseTransformer(verbose=2, logger="log")
    base.log("test", 1)
    cls.return_value.info.assert_called()


# Test save ======================================================== >>

def test_file_is_saved():
    """Assert that the pickle file is saved."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save("auto")
    assert glob.glob("ATOMClassifier")


@patch("atom.basetransformer.pickle")
def test_save_data_false(cls):
    """Assert that the dataset is restored after saving with save_data=False"""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.save(filename="atom", save_data=False)
    assert atom.dataset is not None  # Dataset is restored after saving
    assert atom.holdout is not None  # Holdout is restored after saving
