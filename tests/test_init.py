# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the __main__ method of the ATOM class.

"""

# Import packages
import pytest
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.datasets import load_boston, load_wine, load_breast_cancer
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Functions ====================== >>

def load_df(dataset):
    """ Load dataset as pd.DataFrame """

    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ['target'])
    data = pd.DataFrame(data, columns=columns)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y


# << ====================== Variables ===================== >>

X_dim4 = [[2, 0, 1], [2, 3, 0], [5, 2, 0], [8, 9, 1]]
y_dim4 = ['y', 'n', 'y', 'n']
X_bin, y_bin = load_df(load_breast_cancer())
X_class, y_class = load_df(load_wine())
X_reg, y_reg = load_df(load_boston())


# << ======================= Tests ========================= >>

# << ============== Test handling input data =============== >>

def test_X_y_equal_length():
    """ Assert that error is raised when X and y don't have equal length """

    y = [0, 0, 1, 1, 0]
    pytest.raises(ValueError, ATOMClassifier, X_dim4, y)


def test_y_is1dimensional():
    """ Assert that error is raised when y is not 1-dimensional """

    y = [[0, 0], [1, 1], [0, 1], [1, 1]]
    pytest.raises(ValueError, ATOMClassifier, X_dim4, y)


def test_merger_X_y():
    """ Assert that merger between X and y was successfull """

    atom = ATOMClassifier(X_bin, y_bin)
    merger = X_bin.merge(y_bin.to_frame(), left_index=True, right_index=True)

    # Order of rows can be different
    df1 = merger.sort_values(by=merger.columns.tolist())
    df1.reset_index(drop=True, inplace=True)
    df2 = atom.dataset.sort_values(by=atom.dataset.columns.tolist())
    df2.reset_index(drop=True, inplace=True)
    assert df1.equals(df2)


def test_target_when_y_is_none():
    """ Assert that target is assigned correctly when y is None """

    X, y = X_bin.copy(), y_bin.copy()
    X['target'] = y  # Place y as last column of X
    atom = ATOMClassifier(X)
    assert atom.dataset.columns[-1], atom.target


def test_target_when_last_column():
    """ Assert that target is assigned correctly when last column """

    atom = ATOMClassifier(X_bin, y_bin)
    assert atom.target == y_bin.name
    assert atom.dataset.columns[-1] == atom.target


def test_target_when_y_not_last():
    """ Assert that target is assigned correctly when not last column """

    # When it's not last, it should also move to the last position
    atom = ATOMClassifier(X_bin, y='mean texture')
    assert atom.target == 'mean texture'
    assert atom.dataset.columns[-1] == atom.target


def test_y_in_X():
    """ Assert that the target column given by y is in X """

    pytest.raises(ValueError, ATOMClassifier, X_bin, y='test')


# << ====================== Test parameters ====================== >>

def test_percentage_parameter():
    """ Assert that the percentage parameter is set correctly """

    for p in [0, -1, 120]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, percentage=p)


def test_test_size_parameter():
    """ Assert that the test_size parameter is set correctly """

    for size in [0., -3.1, 12.2]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, test_size=size)


def test_log_parameter():
    """ Assert that the log parameter is set correctly """

    atom = ATOMClassifier(X_bin, y_bin, log='logger')
    atom.outliers()
    assert 1 == 1

    atom = ATOMClassifier(X_bin, y_bin, log='auto')
    atom.outliers()
    assert 2 == 2


def test_verbose_parameter():
    """ Assert that the verbose parameter is set correctly """

    for vb in [-2, 4]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, verbose=vb)


def test_random_state_parameter():
    """ Assert that the random_state parameter is set correctly and works """

    # Check if it gives the same results every time
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom.pipeline(models=['lr', 'lgb', 'pa'],
                  metric='f1',
                  max_iter=3,
                  cv=1,
                  bagging=0)
    atom2 = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom2.pipeline(models=['lr', 'lgb', 'pa'],
                   metric='f1',
                   max_iter=3,
                   cv=1,
                   bagging=0)
    assert atom.lr.score_test == atom2.lr.score_test
    assert atom.lgb.score_test == atom2.lgb.score_test
    assert atom.pa.score_test == atom2.pa.score_test


def test_njobs_parameter():
    """ Assert that the n_jobs parameter is set correctly """

    pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, n_jobs=-200)

    n_cores = multiprocessing.cpu_count()
    for n_jobs in [59, -1, -2, 0]:
        atom = ATOMClassifier(X_bin, y_bin, n_jobs=n_jobs)
        assert 0 < atom.n_jobs <= n_cores


def test_is_fitted_attribute():
    """ Assert that the _is_fitted attribute is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    assert not atom._is_fitted
    atom.pipeline('LR', 'f1', max_iter=0, bagging=0)
    assert atom._is_fitted


def test_is_scaled_attribute():
    """ Assert that the _is_scaled attribute is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    assert not atom._is_scaled
    atom.scale()
    assert atom._is_scaled

    atom2 = ATOMClassifier(atom.X, atom.y)
    assert atom2._is_scaled


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
    len_ = len(X)  # Save number of non-duplicate rows
    y[0], y[21] = np.NaN, np.NaN  # Set NaN to target column for 2 rows
    atom = ATOMClassifier(X, y)
    assert len_ == len(atom.dataset) + 2


# << ==================== Test task assigning ==================== >>

def test_task_assigning():
    """ Assert that self.task is assigned correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    assert atom.task == 'binary classification'

    atom = ATOMClassifier(X_class, y_class)
    assert atom.task == 'multiclass classification'

    atom = ATOMRegressor(X_reg, y_reg)
    assert atom.task == 'regression'


# << ================ Test mapping target column ================ >>

def test_encode_target_column():
    """ Test the encoding of the target column """

    atom = ATOMClassifier(X_dim4, y_dim4)
    assert atom.dataset[atom.target].dtype.kind == 'i'


def test_target_mapping():
    """ Assert that target_mapping attribute is set correctly """

    atom = ATOMClassifier(X_dim4, y_dim4)
    assert atom.mapping == dict(n=0, y=1)

    atom = ATOMClassifier(X_class, y_class)
    assert atom.mapping == {'0': 0, '1': 1, '2': 2}

    atom = ATOMRegressor(X_reg, y_reg)
    assert isinstance(atom.mapping, str)
