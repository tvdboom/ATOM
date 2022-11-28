# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for branch.py

"""

import pandas as pd
import pytest

from atom import ATOMClassifier, ATOMRegressor
from atom.utils import merge

from .conftest import X_bin, X_bin_array, X_class, y_bin, y_bin_array


# Test magic methods =============================================== >>

def test_init_pipeline_to_empty_series():
    """Assert that when starting atom, the estimators are empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.pipeline.empty


def test_init_attrs_are_passed():
    """Assert that the attributes from the parent are passed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance()
    atom.branch = "b2"
    assert atom.b2._idx is not atom.master._idx
    assert atom.b2.adasyn is atom.master.adasyn


def test_delete_current():
    """Assert that we can delete the current branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    del atom.branch
    assert "b2" not in atom._branches


def test_delete_last_branch():
    """Assert that an error is raised when the last branch is deleted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(PermissionError, match=".*last branch.*"):
        del atom.branch


def test_delete_depending_models():
    """Assert that dependent models are deleted with the branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    atom.run("LR")
    del atom.branch
    assert not atom.models


def test_delete_last_og_branch():
    """Assert that an og branch is created if the last one is deleted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    atom.scale()
    assert "og" not in atom._branches
    assert len(atom._get_og_branches()) == 1  # master is the last og branch
    atom.branch = "master"
    del atom.branch
    assert "og" in atom._branches


def test_delete_not_current():
    """Assert that we can delete any branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    assert "b2" in atom._branches
    del atom.branch
    assert "b2" not in atom._branches


def test_repr():
    """Assert that the __repr__  method returns the list of available branches."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert str(atom.branch).startswith("Branch: master\n --> Pipeline")


# Test name property =============================================== >>

def test_name_empty_name():
    """Assert that an error is raised when name is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*can't have an empty name!.*"):
        atom.branch.name = ""


def test_name_existing_name():
    """Assert that an error is raised when name already exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    with pytest.raises(ValueError, match=".*already exists!.*"):
        atom.branch.name = "master"


def test_name_model_name():
    """Assert that an error is raised when name is a model's acronym."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*model's acronym.*"):
        atom.branch.name = "Lda"


def test_name_method():
    """Assert that the branch name changes correctly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch.name = "b1"
    assert atom.branch.name == "b1"
    assert atom.branch.pipeline.name == "b1"


# Test status ====================================================== >>

def test_status_method():
    """Assert that the status method prints the estimators without errors."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.impute()
    atom.branch.status()
    assert str(atom.branch).endswith("\n --> Models: None")


# Test data properties ============================================= >>

def test_dataset_property():
    """Assert that the dataset property returns the data in the branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.dataset is atom.branch._data


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
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    nrows, ncols = int((1 - test_size) * len(X_bin)) + 1, X_bin.shape[1]
    assert atom.branch.X_train.shape == (nrows, ncols)


def test_X_test_property():
    """Assert that the X_test property returns the test feature set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.branch.X_test.shape == (int(test_size * len(X_bin)), X_bin.shape[1])


def test_y_train_property():
    """Assert that the y_train property returns the training target column."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.branch.y_train.shape == (int((1 - test_size) * len(X_bin)) + 1,)


def test_y_test_property():
    """Assert that the y_test property returns the training target column."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.branch.y_test.shape == (int(test_size * len(X_bin)),)


def test_shape_property():
    """Assert that the shape property returns the shape of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.branch.shape == (len(X_bin), X_bin.shape[1] + 1)


def test_columns_property():
    """Assert that the columns property returns the columns of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert list(atom.branch.columns) == list(X_bin.columns) + [y_bin.name]


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


# Test property setters ============================================ >>

def test_setter_with_models():
    """Assert that an error is raised when there are models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    with pytest.raises(PermissionError, match=".*not allowed to change the data.*"):
        atom.X = X_class


def test_dataset_setter():
    """Assert that the dataset setter changes the whole dataset."""
    new_dataset = merge(X_bin, y_bin)
    new_dataset.iat[0, 3] = 4  # Change one value

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.dataset = new_dataset
    assert atom.dataset.iat[0, 3] == 4  # Check the value is changed


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
    assert atom.y[0] == 0  # First value is 1 in original
    atom.y = [1] + list(y_bin.values[1:])
    assert atom.y[0] == 1  # First value changed to 0


def test_X_train_setter():
    """Assert that the X_train setter changes the training feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_train = atom.X_train
    new_X_train.iat[0, 0] = 999
    atom.X_train = new_X_train.to_numpy()  # To numpy to test dtypes are maintained
    assert atom.X_train.iat[0, 0] == 999
    assert list(atom.X_train.dtypes) == list(atom.X_test.dtypes)


def test_X_test_setter():
    """Assert that the X_test setter changes the test feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_test = atom.X_test
    new_X_test.iat[0, 0] = 999
    atom.X_test = new_X_test
    assert atom.X_test.iat[0, 0] == 999


def test_y_train_setter():
    """Assert that the y_train setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_train.iat[0] == 0  # First value is 1 in original
    atom.y_train = [1] + list(atom.y_train.values[1:])
    assert atom.y_train.iat[0] == 1  # First value changed to 0


def test_y_test_setter():
    """Assert that the y_test setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_test.iat[0] == 1  # First value is 0 in original
    atom.y_test = [0] + list(atom.y_test[1:])
    assert atom.y_test.iat[0] == 0  # First value changed to 1


def test_data_properties_to_df():
    """Assert that the data attributes are converted to a df at setter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X = X_bin_array
    assert isinstance(atom.X, pd.DataFrame)


def test_data_properties_to_series():
    """Assert that the data attributes are converted to a series at setter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.y = y_bin_array
    assert isinstance(atom.y, pd.Series)


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
    with pytest.raises(ValueError, match="number of columns"):
        new_X = atom.train
        new_X.insert(0, "new_column", 1)
        atom.train = new_X


def test_setter_error_unequal_column_names():
    """Assert that an error is raised with different column names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match="the same columns"):
        new_X = atom.train.drop(atom.train.columns[0], axis=1)
        new_X.insert(0, "new_column", 1)
        atom.train = new_X


def test_setter_error_unequal_target_names():
    """Assert that an error is raised with different target names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match="the same name"):
        new_y_train = atom.y_train
        new_y_train.name = "different_name"
        atom.y_train = new_y_train
