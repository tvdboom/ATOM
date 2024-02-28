"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for the data module.

"""
import glob
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from atom import ATOMClassifier, ATOMRegressor
from atom.data import Branch, BranchManager
from atom.training import DirectClassifier
from atom.utils.utils import merge

from .conftest import (
    X10, X10_str, X_bin, X_bin_array, X_class, X_idx, y10, y10_str, y_bin,
    y_bin_array, y_idx, y_multiclass,
)


# Test Branch ====================================================== >>

def test_init_empty_pipeline():
    """Assert that an empty branch has an empty pipeline."""
    branch = Branch(name="main")
    assert branch.pipeline.steps == []


def test_location_is_assigned():
    """Assert that a memory location is assigned."""
    branch = Branch(name="main", memory="")
    assert isinstance(branch._location, Path)


def test_branch_repr():
    """Assert that the __repr__ method returns the branch's name."""
    branch = Branch(name="main")
    assert str(branch) == "Branch(main)"


def test_data_property():
    """Assert that the _data property returns the data container."""
    atom = ATOMClassifier(X_bin, y_bin, memory=True, random_state=1)
    atom.branch = "b2"
    assert atom._branches["main"]._container is None  # Inactive branch
    assert atom._branches["main"]._data is not None  # Loads the data


def test_data_property_unassigned_data():
    """Assert that an error is raised when the data is still unassigned."""
    trainer = DirectClassifier("LR")
    with pytest.raises(AttributeError, match=".*no dataset assigned.*"):
        print(trainer.dataset)


def test_name_empty_name():
    """Assert that an error is raised when name is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*can't have an empty name.*"):
        atom.branch.name = ""


def test_name_ensemble_name():
    """Assert that an error is raised when name is the name of an ensemble."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*can't begin with 'stack'.*"):
        atom.branch.name = "stacked"


def test_name_model_name():
    """Assert that an error is raised when name is a model's acronym."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*model's acronym.*"):
        atom.branch.name = "Lda"


def test_name_setter():
    """Assert that the branch name changes correctly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch.name = "b1"
    assert atom.branch.name == "b1"


def test_pipeline_property():
    """Assert that the pipeline property returns the current pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert len(atom.branch.pipeline) == 1


def test_mapping_property():
    """Assert that the dataset property returns the target's mapping."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.mapping == {}


def test_dataset_property():
    """Assert that the dataset property returns the data in the branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.dataset is atom.branch._data.data


def test_train_property():
    """Assert that the train property returns the training set."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    n_rows, ncols = int((1 - 0.3) * len(X_bin)) + 1, X_bin.shape[1] + 1
    assert atom.branch.train.shape == (n_rows, ncols)


def test_test_property():
    """Assert that the test property returns the test set."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    assert atom.branch.test.shape == (int(0.3 * len(X_bin)), X_bin.shape[1] + 1)


def test_holdout_property():
    """Assert that the holdout property returns a transformed holdout set."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.scale()
    assert not atom.holdout.equals(atom.branch.holdout)


def test_X_property():
    """Assert that the X property returns the feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.X.shape == (len(X_bin), X_bin.shape[1])


def test_y_property():
    """Assert that the y property returns the target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.y.shape == (len(y_bin),)


def test_X_train_property():
    """Assert that the X_train property returns the training feature set."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    nrows, ncols = int((1 - 0.3) * len(X_bin)) + 1, X_bin.shape[1]
    assert atom.branch.X_train.shape == (nrows, ncols)


def test_X_test_property():
    """Assert that the X_test property returns the test feature set."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    assert atom.branch.X_test.shape == (int(0.3 * len(X_bin)), X_bin.shape[1])


def test_y_train_property():
    """Assert that the y_train property returns the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    assert atom.branch.y_train.shape == (int((1 - 0.3) * len(X_bin)) + 1,)


def test_y_test_property():
    """Assert that the y_test property returns the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    assert atom.branch.y_test.shape == (int(0.3 * len(X_bin)),)


def test_shape_property():
    """Assert that the shape property returns the shape of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.shape == (len(X_bin), X_bin.shape[1] + 1)


def test_columns_property():
    """Assert that the columns property returns the columns of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert list(atom.branch.columns) == [*X_bin.columns, y_bin.name]


def test_n_columns_property():
    """Assert that the n_columns property returns the number of columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.n_columns == len(X_bin.columns) + 1


def test_features_property():
    """Assert that the features property returns the features of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert list(atom.branch.features) == list(X_bin.columns)


def test_n_features_property():
    """Assert that the n_features property returns the number of features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.n_features == len(X_bin.columns)


def test_target_property():
    """Assert that the target property returns the last column in the dataset."""
    atom = ATOMRegressor(X_bin, "mean radius", random_state=1)
    assert atom.branch.target == "mean radius"


def test_all_property():
    """Assert that the _all property returns the dataset + holdout."""
    atom = ATOMRegressor(X_bin, y_bin, holdout_size=0.1, random_state=1)
    assert len(atom.branch.dataset) != len(X_bin)
    assert len(atom.branch._all) == len(X_bin)


def test_dataset_setter():
    """Assert that the dataset setter changes the whole dataset."""
    new_dataset = merge(X_bin, y_bin)
    new_dataset.iloc[0, 3] = 4  # Change one value

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.dataset = new_dataset
    assert atom.dataset.iloc[0, 3] == 4  # Check the value is changed


def test_train_setter():
    """Assert that the train setter changes the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train = atom.train.iloc[:100, :]
    assert atom.train.shape == (100, X_bin.shape[1] + 1)


def test_test_setter():
    """Assert that the test setter changes the test set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.test = atom.test.iloc[:100, :]
    assert atom.test.shape == (100, X_bin.shape[1] + 1)


def test_X_setter():
    """Assert that the X setter changes the feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X = atom.X.iloc[:, :10]
    assert atom.X.shape == (len(X_bin), 10)


def test_y_setter():
    """Assert that the y setter changes the target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y[0] == 0  # The first value is 1 in original
    atom.y = [1, *y_bin.values[1:]]
    assert atom.y[0] == 1  # First value changed to 0


def test_X_train_setter():
    """Assert that the X_train setter changes the training feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_train = atom.X_train
    new_X_train.iloc[0, 0] = 999
    atom.X_train = new_X_train.to_numpy()  # To numpy to test dtypes are maintained
    assert atom.X_train.iloc[0, 0] == 999
    assert list(atom.X_train.dtypes) == list(atom.X_test.dtypes)


def test_X_test_setter():
    """Assert that the X_test setter changes the test feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_test = atom.X_test
    new_X_test.iloc[0, 0] = 999
    atom.X_test = new_X_test
    assert atom.X_test.iloc[0, 0] == 999


def test_y_train_setter():
    """Assert that the y_train setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_train.iloc[0] == 0  # The first value is 1 in original
    atom.y_train = [1, *atom.y_train[1:]]
    assert atom.y_train.iloc[0] == 1  # First value changed to 0


def test_y_test_setter():
    """Assert that the y_test setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_test.iloc[0] == 1  # The first value is 0 in original
    atom.y_test = [0, *atom.y_test[1:]]
    assert atom.y_test.iloc[0] == 0  # First value changed to 1


def test_data_properties_to_df():
    """Assert that the data attributes are converted to a df at setter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X = X_bin_array
    assert isinstance(atom.branch.X, pd.DataFrame)


def test_data_properties_to_series():
    """Assert that the data attributes are converted to a series at setter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.y = y_bin_array
    assert isinstance(atom.branch.y, pd.Series)


def test_setter_error_unequal_rows():
    """Assert that an error is raised when the setter has unequal rows."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match="number of rows"):
        atom.X_train = X_bin


def test_setter_error_unequal_index():
    """Assert that an error is raised when the setter has unequal indices."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match="the same indices"):
        atom.y = pd.Series(y_bin_array, index=range(10, len(y_bin_array) + 10))


def test_setter_error_unequal_columns():
    """Assert that an error is raised when the setter has unequal columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X = atom.train
    new_X["new_column"] = 1
    with pytest.raises(ValueError, match="number of columns"):
        atom.train = new_X


def test_setter_error_unequal_column_names():
    """Assert that an error is raised with different column names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X = atom.train.drop(columns=atom.train.columns[0])
    new_X.insert(0, "new_column", 1)
    with pytest.raises(ValueError, match="the same columns"):
        atom.train = new_X


def test_setter_error_unequal_target_names():
    """Assert that an error is raised with different target names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_y_train = atom.y_train
    new_y_train.name = "different_name"
    with pytest.raises(ValueError, match="the same name"):
        atom.y_train = new_y_train


def test_get_rows_is_dataframe():
    """Assert that a dataframe returns the rows directly."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    assert len(atom.branch._get_rows(rows=atom.test)) == len(atom.test)


def test_get_rows_is_index():
    """Assert that an Index object returns the rows directly."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    assert len(atom.branch._get_rows(rows=atom.test.index)) == len(atom.test)


def test_get_rows_is_range():
    """Assert that a range of rows is returned."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    assert len(atom.branch._get_rows(rows=range(20, 40))) == 20


def test_get_rows_is_slice():
    """Assert that a slice of rows is returned."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    assert len(atom.branch._get_rows(rows=slice(20, 100, 2))) == 40


def test_get_rows_by_exact_match():
    """Assert that name can select a row."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    assert atom.branch._get_rows(rows="index_23").index[0] == "index_23"


def test_get_rows_by_int():
    """Assert that rows can be retrieved by their index position."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    with pytest.raises(IndexError, match=".*out of range.*"):
        atom.branch._get_rows(rows=1000)
    assert atom.branch._get_rows(rows=100).equals(atom.dataset.iloc[[100]])


def test_get_rows_by_str():
    """Assert that rows can be retrieved by name, data set or regex."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    assert len(atom.branch._get_rows(rows="index_34+index_58")) == 2
    assert len(atom.branch._get_rows(rows=["index_34+index_58", "index_57"])) == 3
    assert len(atom.branch._get_rows(rows="test")) == len(atom.test)
    with pytest.raises(ValueError, match=".*No holdout data set was declared.*"):
        atom.branch._get_rows(rows="holdout")
    assert len(atom.branch._get_rows(rows="index_3.*")) == 111
    assert len(atom.branch._get_rows(rows="!index_3")) == len(X_idx) - 1
    assert len(atom.branch._get_rows(rows="!index_3.*")) == len(X_idx) - 111


def test_get_rows_none_selected():
    """Assert that an error is raised when no rows are selected."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    with pytest.raises(ValueError, match=".*No rows were selected.*"):
        atom.branch._get_rows(rows=slice(1000, 2000))


def test_get_rows_include_or_exclude():
    """Assert that an error is raised when rows are included and excluded."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, random_state=1)
    with pytest.raises(ValueError, match=".*either include or exclude rows.*"):
        atom.branch._get_rows(rows=["index_34", "!index_36"])


def test_get_columns_is_None():
    """Assert that all columns are returned."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert len(atom.branch._get_columns(columns=None)) == 5
    assert len(atom.branch._get_columns(columns=None, only_numerical=True)) == 4
    assert len(atom.branch._get_columns(columns=None, include_target=False)) == 4


def test_get_columns_by_dataframe():
    """Assert that a dataframe retrieves columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert len(atom.branch._get_columns(columns=atom.X_train)) == X_bin.shape[1]


def test_get_columns_by_segment():
    """Assert that a range or slice retrieves columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert len(atom.branch._get_columns(columns=slice(2, 6))) == 4
    assert len(atom.branch._get_columns(columns=range(2, 6))) == 4


def test_get_columns_by_int():
    """Assert that an index can retrieve columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(IndexError, match=".*out of range for data.*"):
        atom.branch._get_columns(columns=40)
    assert atom.branch._get_columns(columns=0) == ["mean radius"]


def test_get_columns_by_str():
    """Assert that columns can be retrieved by name or regex."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert len(atom.branch._get_columns("mean radius+mean texture")) == 2
    assert len(atom.branch._get_columns(["mean radius+mean texture", "mean area"])) == 3
    assert len(atom.branch._get_columns("mean .*")) == 10
    assert len(atom.branch._get_columns("!mean radius")) == X_bin.shape[1]
    assert len(atom.branch._get_columns("!mean .*")) == X_bin.shape[1] - 9
    with pytest.raises(ValueError, match=".*any column that matches.*"):
        atom.branch._get_columns("invalid")


def test_get_columns_by_type():
    """Assert that columns can be retrieved by type."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert len(atom.branch._get_columns(columns="number")) == 4
    assert len(atom.branch._get_columns(columns="!number")) == 1


def test_get_columns_exclude():
    """Assert that columns can be excluded using `!`."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*not find any column.*"):
        atom.branch._get_columns(columns="!invalid")
    assert len(atom.branch._get_columns(columns="!mean radius")) == 30
    assert len(atom.branch._get_columns(columns=["!mean radius", "!mean texture"])) == 29


def test_get_columns_none_selected():
    """Assert that an error is raised when no columns are selected."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*At least one column.*"):
        atom.branch._get_columns(columns="datetime")


def test_get_columns_include_or_exclude():
    """Assert that an error is raised when cols are included and excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*either include or exclude columns.*"):
        atom.branch._get_columns(columns=["mean radius", "!mean texture"])


def test_get_columns_remove_duplicates():
    """Assert that duplicate columns are ignored."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch._get_columns(columns=[0, 1, 0]) == ["mean radius", "mean texture"]


def test_get_target_column():
    """Assert that the target column can be retrieved."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    assert atom.branch._get_target(target="c", only_columns=True) == "c"
    assert atom.branch._get_target(target=1, only_columns=True) == "b"


def test_get_target_column_str_invalid():
    """Assert that an error is raised when the column is invalid."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    with pytest.raises(ValueError, match=".*is not one of the target columns.*"):
        atom.branch._get_target(target="invalid", only_columns=True)


def test_get_target_column_int_invalid():
    """Assert that an error is raised when the column is invalid."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    with pytest.raises(ValueError, match=".*There are 3 target columns.*"):
        atom.branch._get_target(target=3, only_columns=True)


def test_get_target_class():
    """Assert that the target class can be retrieved."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    atom.clean()
    assert atom.branch._get_target(target="y")[1] == 1
    assert atom.branch._get_target(target=0)[1] == 0


def test_get_target_class_str_invalid():
    """Assert that an error is raised when the target is invalid."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    with pytest.raises(ValueError, match=".*not found in the mapping.*"):
        atom.branch._get_target(target="invalid")


def test_get_target_class_int_invalid():
    """Assert that an error is raised when the value is invalid."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    with pytest.raises(ValueError, match=".*There are 2 classes.*"):
        atom.branch._get_target(target=3)


def test_get_target_tuple_no_multioutput():
    """Assert that the target class can be retrieved."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    with pytest.raises(ValueError, match=".*only accepted for multioutput tasks.*"):
        atom.branch._get_target(target=(2, 1))


def test_get_target_tuple():
    """Assert that the target column and class can be retrieved."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    assert atom.branch._get_target(target=(2,)) == (2, 0)
    assert atom.branch._get_target(target=("a", 2)) == (0, 2)


def test_get_target_tuple_invalid_length():
    """Assert that the target class can be retrieved."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    with pytest.raises(ValueError, match=".*a tuple of length 2.*"):
        atom.branch._get_target(target=(2, 1, 2))


def test_load_store():
    """Assert that inactive branches are stored and loaded from memory."""
    atom = ATOMClassifier(X_bin, y=y_bin, memory="", random_state=1)
    atom.branch = "2"
    assert atom._branches["main"]._container is None  # Stored
    assert atom._branches["main"].load(assign=False) is not None  # Loaded
    atom._branches["main"].load(assign=True)
    assert atom._branches["main"]._container is not None


def test_load_no_file():
    """Assert that an error is raised when trying to load without a file."""
    atom = ATOMClassifier(X_bin, y=y_bin, memory="", random_state=1)
    atom.branch = "2"
    os.remove(atom.branch._location.joinpath("Branch(main).pkl"))
    with pytest.raises(FileNotFoundError, match=".*no data stored.*"):
        atom.branch = "main"


def test_load_no_dir():
    """Assert that an error is raised when trying to load without a directory."""
    atom = ATOMClassifier(X_bin, y=y_bin, memory="", random_state=1)
    atom.branch = "2"
    atom.memory.clear()
    with pytest.raises(FileNotFoundError, match=".*does not exist.*"):
        atom.branch = "main"


def test_check_scaling_scaler_in_pipeline():
    """Assert that check_scaling returns True when there's a scaler in the pipeline."""
    atom = ATOMClassifier(X_bin, y=y_bin, random_state=1)
    assert not atom.branch.check_scaling()
    atom.add(MinMaxScaler())
    assert atom.branch.check_scaling()


def test_check_scaling():
    """Assert that the check_scaling method returns whether the data is scaled."""
    scaler = StandardScaler()
    scaler.__class__.__name__ = "OtherName"

    atom = ATOMClassifier(X_bin, y=y_bin, random_state=1)
    atom.add(scaler)
    assert atom.branch.check_scaling()


def test_check_scaling_drop_binary():
    """Assert that binary rows are dropped to check scaling."""
    atom = ATOMClassifier(np.tile(y10, (10, 1)), y=y10, random_state=1)
    assert atom.branch.check_scaling()


# Test BranchManager =============================================== >>

def test_branchmanager_repr():
    """Assert that the __repr__ method returns the branches."""
    assert str(BranchManager()) == "BranchManager([main], og=main)"


def test_branchmanager_len():
    """Assert that the __len__ method returns the number of branches."""
    assert len(BranchManager()) == 1


def test_branchmanager_iter():
    """Assert that the __iter__ method iterates over the branches."""
    assert str(next(iter(BranchManager()))) == "Branch(main)"


def test_branchmanager_contains():
    """Assert that the __contains__ method checks if there is a branch."""
    assert "main" in BranchManager()


def test_branchmanager_getitem():
    """Assert that the __getitem__ method returns a branch."""
    manager = BranchManager()
    assert manager[0].name == "main"
    with pytest.raises(IndexError, match=".*has no branch.*"):
        print(manager["invalid"])


def test_og():
    """Assert that the og property returns the original branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom._branches.og is atom._branches["main"]
    atom.scale()
    assert atom._branches.og is atom._branches._og


def test_current():
    """Assert that the current property returns the active branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom._branches.current is atom._branches["main"]
    atom.branch = "b2"
    assert atom._branches.current is atom._branches["b2"]


def test_current_setter():
    """Assert that the current property returns the active branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom._branches.current is atom._branches["main"]
    atom.branch = "b2"
    assert atom._branches.current is atom._branches["b2"]


def test_add_og():
    """Assert that the og branch is created separately."""
    atom = ATOMClassifier(X_bin, y_bin, memory="", random_state=1)
    atom.scale()
    assert atom._branches.og.name == "og"
    assert atom._branches.og._container is None
    assert atom._branches.og._data is not None
    assert glob.glob("joblib/atom/Branch(og).pkl")


def test_add_store_previous_branch():
    """Assert that old branches are stored when inactive."""
    atom = ATOMClassifier(X_bin, y_bin, memory="", random_state=1)
    atom.scale()
    atom.branch = "b2"
    assert atom._branches["main"]._container is None
    assert atom._branches.current is atom._branches["b2"]


def test_add_copy_from_parent():
    """Assert that the parent's data is passed to the new branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    atom.branch = "b2"
    assert_frame_equal(atom.dataset, atom._branches["main"].dataset)
    assert atom.pipeline is not atom._branches["main"].pipeline
    assert atom.pipeline.steps[0][1] is atom._branches["main"].pipeline.steps[0][1]
    assert atom.branch._mapping is not atom._branches["main"]._mapping
    assert atom.standard_ is atom._branches["main"].standard_


def test_holdout_is_same():
    """Assert that the holdout data set is the same across branches."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.branch = "b2"
    assert atom.holdout is atom._branches["b2"].holdout


def test_reset():
    """Assert that the reset methods delete all branches."""
    atom = ATOMClassifier(X_bin, y_bin, memory="", random_state=1)
    atom.scale()
    atom.branch = "b2"
    assert atom.og is not atom.branch
    assert len(atom._branches) == 2
    assert glob.glob("joblib/atom/Branch(main).pkl")
    atom._branches.reset(hard=True)
    assert len(atom._branches) == 1
    assert not glob.glob("joblib/atom/Branch(main).pkl")
    assert atom.og is atom.branch


# Test data engines ================================================ >>

def test_numpy_engine():
    """Assert that the numpy engine returns a numpy array."""
    atom = ATOMClassifier(X_bin, y_bin, engine="numpy", random_state=1)
    assert isinstance(atom.dataset, np.ndarray)


def test_pandas_numpy_engine():
    """Assert that the pandas engine returns numpy dtypes."""
    atom = ATOMClassifier(X_bin, y_bin, engine="pandas", random_state=1)
    assert all(isinstance(dtype, np.dtype) for dtype in atom.dataset.dtypes)
    assert isinstance(atom.y.dtype, np.dtype)


def test_pandas_pyarrow_engine():
    """Assert that the pandas-pyarrow engine returns pyarrow dtypes."""
    atom = ATOMClassifier(X_bin, y_bin, engine="pandas-pyarrow", random_state=1)
    assert all(isinstance(dtype, pd.ArrowDtype) for dtype in atom.dataset.dtypes)
    assert isinstance(atom.y.dtype, pd.ArrowDtype)


def test_polars_engine():
    """Assert that the polars engine returns polars types."""
    atom = ATOMClassifier(X_bin, y_bin, engine="polars", random_state=1)
    assert isinstance(atom.X, pl.DataFrame)
    assert isinstance(atom.y, pl.Series)


def test_polars_lazy_engine():
    """Assert that the polars-lazy engine returns polars types."""
    atom = ATOMClassifier(X_bin, y_bin, engine="polars-lazy", random_state=1)
    assert isinstance(atom.X, pl.LazyFrame)
    assert isinstance(atom.y, pl.Series)


def test_pyarrow_engine():
    """Assert that the pyarrow engine returns pyarrow types."""
    atom = ATOMClassifier(X_bin, y_bin, engine="pyarrow", random_state=1)
    assert isinstance(atom.X, pa.Table)
    assert isinstance(atom.y, pa.Array)


@patch.dict("sys.modules", {"modin": MagicMock(spec=["__spec__", "pandas"])})
def test_modin_engine():
    """Assert that the modin engine returns modin types."""
    atom = ATOMClassifier(X_bin, y_bin, engine="modin", random_state=1)
    assert "DataFrame" in str(atom.X)
    assert "Series" in str(atom.y)


def test_dask_engine():
    """Assert that the dask engine returns dask types."""
    atom = ATOMClassifier(X_bin, y_bin, engine="dask", random_state=1)
    assert isinstance(atom.X, dd.DataFrame)
    assert isinstance(atom.y, dd.Series)


@patch.dict("sys.modules", {"pyspark.sql": MagicMock(spec=["__spec__", "SparkSession"])})
def test_pyspark_engine():
    """Assert that the pyspark engine returns pyspark types."""
    atom = ATOMClassifier(X_bin, y_bin, engine="pyspark", random_state=1)
    assert "createDataFrame" in str(atom.X)


@patch.dict("sys.modules", {"pyspark": MagicMock(spec=["__spec__", "pandas"])})
def test_pyspark_pandas_engine():
    """Assert that the pyspark-pandas engine returns pyspark pandas types."""
    atom = ATOMClassifier(X_bin, y_bin, engine="pyspark-pandas", random_state=1)
    assert "DataFrame" in str(atom.X)
    assert "Series" in str(atom.y)
