# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the BaseCleaner class.

"""

# Import packages
import pytest
import pandas as pd
from atom import ATOMClassifier, ATOMRegressor
from atom.data_cleaning import BaseCleaner, Scaler, StandardCleaner
from .utils import (
    X_bin, y_bin, X_class, y_class, X_bin_array, y_bin_array, X_dim4
    )


# << ================== Test __init__ ================== >>

def test_kwargs_to_attributes():
    """Assert that kwargs are attached as attributes of the class."""
    estimator = BaseCleaner(task='regression')
    assert estimator.verbose == 0
    assert estimator.task == 'regression'


def test_atom_parameters_to_attributes():
    """Assert that atom attributes are (in second place) attached as attrs."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1)
    estimator = BaseCleaner(verbose=3, atom=atom)
    assert estimator.verbose == 3
    assert atom.task.startswith('binary')


def test_removal_atom_kwarg():
    """Assert that the atom kwarg is removed from the kwargs attribute."""
    atom = ATOMClassifier(X_bin, y_bin)
    estimator = BaseCleaner(atom=atom)
    assert not estimator.kwargs.get('atom')


# << ================= Test prepare_input ================ >>

def test_copy_input_data():
    """Assert that the prepare_input method uses copies of the data."""
    X, y = BaseCleaner().prepare_input(X_bin, y_bin)
    assert X is not X_bin
    assert y is not y_bin


def test_to_pandas():
    """Assert that the data provided is converted to pandas objects."""
    X, y = BaseCleaner.prepare_input(X_bin_array, y_bin_array)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_reset_indices():
    """Assert that the index is reset for both X and y."""
    # Create dataset with new indices (starting from 10)
    X = X_bin.set_index(pd.Index(range(10, len(X_bin)+10)))
    y = y_bin.reindex(pd.Index(range(10, len(X_bin)+10)))
    X, y = BaseCleaner.prepare_input(X, y)
    assert X.index[0] == 0 and X.index[-1] == len(X) - 1
    assert y.index[0] == 0 and y.index[-1] == len(X) - 1


def test_equal_length():
    """Assert that an error is raised when X and y don't have equal length."""
    pytest.raises(ValueError, BaseCleaner.prepare_input, X_dim4, [0, 1, 1])


def test_y_is1dimensional():
    """Assert that an error is raised when y is not 1-dimensional."""
    y = [[0, 0], [1, 1], [0, 1], [1, 1]]
    pytest.raises(ValueError, BaseCleaner.prepare_input, X_dim4, y)


def test_target_is_string():
    """Assert that the target column is assigned correctly for a string."""
    _, y = BaseCleaner.prepare_input(X_bin, y='mean radius')
    assert y.name == 'mean radius'


def test_target_not_in_dataset():
    """Assert that the target column given by y is in X."""
    pytest.raises(ValueError, BaseCleaner.prepare_input, X_bin, 'X')


def test_target_is_int():
    """Assert that target column is assigned correctly for an integer."""
    _, y = BaseCleaner.prepare_input(X_bin, y=0)
    assert y.name == 'mean radius'


def test_target_is_none():
    """Assert that target column stays None when empty input."""
    _, y = BaseCleaner.prepare_input(X_bin, y=None)
    assert y is None


# << ================= Test fit_transform ================ >>

def test_fit_transform():
    """Assert that the fit_transform method works as intended."""
    X = X_bin.copy()
    X['test_column'] = 1  # Create a column with minimum cardinality
    X_1 = Scaler().fit_transform(X)
    X_2 = Scaler().fit(X_bin).transform(X)
    assert X_1.equals(X_2)


def test_fit_transform_no_fit():
    """Assert that the fit_transform method works when no fit method."""
    X = X_bin.copy()
    X['test_column'] = 1  # Create a column with minimum cardinality
    X_1 = StandardCleaner().fit_transform(X)
    X_2 = StandardCleaner().transform(X)
    assert X_1.equals(X_2)


# << ================ Test mapping target column ================ >>

def test_label_encoder_target_column():
    """ Assert that the label-encoder for the target column works """

    atom = ATOMClassifier(X_dim4, y_dim4, random_state=1)
    assert np.all((atom.y == 0) | (atom.y == 1))


def test_target_mapping():
    """ Assert that target_mapping attribute is set correctly """

    atom = ATOMClassifier(X_dim4, y_dim4, random_state=1)
    assert atom.mapping == dict(n=0, y=1)

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.mapping == {'0.0': 0, '1.0': 1, '2.0': 2}


# << ==================== Test data cleaning ==================== >>

def test_remove_invalid_column_type():
    """ Assert that self.dataset removes invalid columns types """

    X, y = X_bin.copy(), y_bin.copy()
    X['invalid_column'] = pd.to_datetime(X['mean radius'])  # Datetime column
    atom = ATOMClassifier(X, y)
    assert 'invalid_column' not in atom.dataset.columns


def test_remove_maximum_cardinality():
    """ Assert that self.dataset removes columns with maximum cardinality """

    X, y = X_bin.copy(), y_bin.copy()
    # Create column with all different values
    X['invalid_column'] = [str(i) for i in range(len(X))]
    atom = ATOMClassifier(X, y)
    assert 'invalid_column' not in atom.dataset.columns


def test_raise_one_target_value():
    """ Assert that error raises when there is only 1 target value """

    X, y = X_bin.copy(), y_bin.copy()
    y = [1 for _ in range(len(y))]  # All targets are equal to 1
    pytest.raises(ValueError, ATOMClassifier, X, y)


def test_remove_minimum_cardinality():
    """ Assert that self.dataset removes columns with only 1 value """

    X, y = X_bin.copy(), y_bin.copy()
    # Create column with all different values
    X['invalid_column'] = [2.3 for i in range(len(X))]
    atom = ATOMClassifier(X, y)
    assert 'invalid_column' not in atom.dataset.columns


def test_remove_rows_nan_target():
    """ Assert that self.dataset removes rows with NaN in target column """

    X, y = X_bin.copy(), y_bin.copy()
    length = len(X)  # Save number of rows
    y[0], y[21] = np.NaN, np.NaN  # Set NaN to target column for 2 rows
    atom = ATOMClassifier(X, y)
    assert length == len(atom.dataset) + 2