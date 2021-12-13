# coding: utf-8

"""Automated Tool for Optimized Modelling (ATOM)

Author: Mavs
Description: Unit tests for basetransformer.py

"""

# Standard packages
import glob
import pytest
import pandas as pd
import multiprocessing
from scipy import sparse
from unittest.mock import patch
from pandas.testing import assert_frame_equal

# Own modules
from atom import ATOMClassifier
from atom.training import DirectClassifier
from atom.basetransformer import BaseTransformer
from atom.utils import merge
from .utils import (
    FILE_DIR, X_bin, y_bin, X_bin_array, y_bin_array, mnist,
    X_text, y_text, X10, y10, bin_train, bin_test,
)


# Test properties ================================================== >>

def test_n_jobs_maximum_cores():
    """Assert that value equals n_cores if maximum is exceeded."""
    base = BaseTransformer(n_jobs=1000)
    assert base.n_jobs == multiprocessing.cpu_count()


@pytest.mark.parametrize("n_jobs", [0, -1000])
def test_n_jobs_invalid(n_jobs):
    """Assert that an error is raised when n_jobs is invalid."""
    with pytest.raises(ValueError, match=r".*n_jobs parameter.*"):
        BaseTransformer(n_jobs=n_jobs)


def test_negative_n_jobs():
    """Assert that value is set correctly for negative values."""
    base = BaseTransformer(n_jobs=-1)
    assert base.n_jobs == multiprocessing.cpu_count()

    base = BaseTransformer(n_jobs=-2)
    assert base.n_jobs == multiprocessing.cpu_count() - 1


@pytest.mark.parametrize("verbose", [-2, 3])
def test_verbose_parameter(verbose):
    """Assert that the verbose parameter is in correct range."""
    with pytest.raises(ValueError, match=r".*verbose parameter.*"):
        BaseTransformer(verbose=verbose)


def test_warnings_parameter_bool():
    """Assert that the warnings parameter works for a bool."""
    base = BaseTransformer(warnings=True)
    assert base.warnings == "default"

    base = BaseTransformer(warnings=False)
    assert base.warnings == "ignore"


def test_warnings_parameter_invalid_str():
    """Assert that an error is raised for an invalid string for warnings."""
    with pytest.raises(ValueError, match=r".*warnings parameter.*"):
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

    BaseTransformer(logger=FILE_DIR + "auto")
    cls.assert_called_once()


@patch("atom.utils.logging.getLogger")
def test_crash_with_logger(cls):
    """Assert that the crash decorator works with a logger."""
    atom = ATOMClassifier(X_bin, y_bin, logger=FILE_DIR + "log")
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
    with pytest.raises(ValueError, match=r".*random_state parameter.*"):
        BaseTransformer(random_state=-1)


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
    atom = ATOMClassifier(*mnist, random_state=1)
    assert atom.X.columns == ["Multidimensional feature"]
    assert atom.X.iloc[0, 0].shape == (28, 28, 1)


def test_text_to_corpus():
    """Assert that for text data the column is named Corpus."""
    atom = ATOMClassifier(X_text, y_text, random_state=1)
    assert atom.X.columns == ["Corpus"]


def test_sparse_matrices():
    """Assert that sparse matrices are accepted as input type."""
    X, y = BaseTransformer._prepare_input(sparse.eye(10), y10)
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)


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


# Test _get_data =================================================== >>

def test_empty_data_arrays():
    """Assert that an error is raised when no data is provided."""
    with pytest.raises(ValueError, match=r".*data arrays are empty.*"):
        ATOMClassifier(n_rows=100, random_state=1)


def test_data_already_set():
    """Assert that if there already is data, the call to run can be empty."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    trainer.run()
    assert trainer.dataset.equals(pd.concat([bin_train, bin_test]))
    assert trainer.branch.idx[0].equals(bin_train.index)
    assert trainer.branch.idx[1].equals(bin_test.index)


def test_input_is_X():
    """Assert that input X works."""
    atom = ATOMClassifier(X_bin, random_state=1)
    assert atom.dataset.shape == X_bin.shape


def test_input_is_X_with_parameter_y():
    """Assert that input X can be combined with parameter y."""
    atom = ATOMClassifier(X_bin, y="mean texture", random_state=1)
    assert atom.target == "mean texture"


def test_input_invalid_holdout():
    """Assert that an error is raised when holdout is invalid."""
    with pytest.raises(ValueError, match=r".*holdout_size parameter.*"):
        ATOMClassifier(X_bin, test_size=0.3, holdout_size=0.8)


@pytest.mark.parametrize("holdout_size", [0.1, 40])
def test_input_is_X_with_holdout(holdout_size):
    """Assert that input X can be combined with a holdout set."""
    atom = ATOMClassifier(X_bin, holdout_size=holdout_size, random_state=1)
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
    with pytest.raises(ValueError, match=r".*n_rows parameter.*"):
        ATOMClassifier(X_bin, y_bin, n_rows=1e6, random_state=1)


def test_no_shuffle_X_y():
    """Assert that the order is kept when shuffle=False."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, n_rows=30)
    assert_frame_equal(atom.X, X_bin.iloc[:30, :])


def test_length_dataset():
    """Assert that the dataset is always len>=2."""
    with pytest.raises(ValueError, match=r".*n_rows parameter.*"):
        ATOMClassifier(X10, y10, n_rows=0.01, random_state=1)


@pytest.mark.parametrize("test_size", [-2, 0, 1000])
def test_test_size_parameter(test_size):
    """Assert that the test_size parameter is in correct range."""
    with pytest.raises(ValueError, match=r".*test_size parameter.*"):
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
    assert trainer.dataset.equals(pd.concat([bin_train, bin_test]))


def test_input_is_train_test():
    """Assert that input train, test works."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.dataset.equals(pd.concat([bin_train, bin_test]))


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
    assert trainer.dataset.equals(pd.concat([bin_train, bin_test]))
    assert trainer.holdout.equals(bin_test)


def test_input_is_train_test_holdout():
    """Assert that input train, test, holdout works."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test, bin_test)
    assert trainer.dataset.equals(pd.concat([bin_train, bin_test]))
    assert trainer.holdout.equals(bin_test)


def test_4_data_provided():
    """Assert that the 4 elements input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(X_train, X_test, y_train, y_test)
    assert trainer.dataset.equals(pd.concat([bin_train, bin_test]))


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
    assert trainer.dataset.equals(pd.concat([bin_train, bin_test]))
    assert trainer.holdout.equals(bin_test)


def test_invalid_input():
    """Assert that an error is raised when input arrays are invalid."""
    trainer = DirectClassifier("LR", random_state=1)
    with pytest.raises(ValueError, match=r".*Invalid data arrays.*"):
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
    with pytest.raises(ValueError, match=r".*has to be <1 when a train and test.*"):
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
    with pytest.raises(ValueError, match=r".*train and test set do not have.*"):
        ATOMClassifier(X10, bin_test, random_state=1)


def test_unequal_columns_holdout():
    """Assert that an error is raised when holdout has different columns."""
    with pytest.raises(ValueError, match=r".*holdout set does not have.*"):
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

@patch("atom.utils.logging.getLogger")
def test_log(cls):
    """Assert the log method works."""
    base = BaseTransformer(verbose=2, logger=FILE_DIR + "log")
    base.log("test", 1)
    cls.return_value.info.assert_called()


# Test save ======================================================== >>

def test_file_is_saved():
    """Assert that the pickle file is saved."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + "auto")
    assert glob.glob(FILE_DIR + "ATOMClassifier")


@patch("atom.basetransformer.dill")
def test_save_data_false(cls):
    """Assert that the dataset is restored after saving with save_data=False"""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.save(filename=FILE_DIR + "atom", save_data=False)
    assert atom.dataset is not None  # Dataset is restored after saving
    assert atom.holdout is not None  # Holdout is restored after saving
