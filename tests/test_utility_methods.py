# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the utility methods of the ATOM class.

"""

# Import packages
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Variables ===================== >>

X_dim4 = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
y_dim4 = [0, 1, 1, 0]
X_bin, y_bin = load_breast_cancer(return_X_y=True)
X_class, y_class = load_wine(return_X_y=True)
X_reg, y_reg = load_boston(return_X_y=True)


# << ================ Test class variables ================= >>

def test_set_style():
    """ Assert that the set_style classmethod works as intended """

    atom = ATOMClassifier(X_dim4, y_dim4)
    atom.set_style('white')
    assert ATOMClassifier.style == 'white'


def test_set_palette():
    """ Assert that the set_palette classmethod works as intended """

    atom = ATOMClassifier(X_dim4, y_dim4)
    atom.set_palette('Blues')
    assert ATOMClassifier.palette == 'Blues'


def test_set_title_fontsize():
    """ Assert that the set_title_fontsize classmethod works as intended """

    atom = ATOMClassifier(X_dim4, y_dim4)
    atom.set_title_fontsize(21)
    assert ATOMClassifier.title_fontsize == 21


def test_set_label_fontsize():
    """ Assert that the set_label_fontsize classmethod works as intended """

    atom = ATOMClassifier(X_dim4, y_dim4)
    atom.set_label_fontsize(4)
    assert ATOMClassifier.label_fontsize == 4


def test_set_tick_fontsize():
    """ Assert that the set_tick_fontsize classmethod works as intended """

    atom = ATOMClassifier(X_dim4, y_dim4)
    atom.set_tick_fontsize(13)
    assert ATOMClassifier.tick_fontsize == 13


# << ================ Test _split_dataset ================== >>

def test_dataset_is_shuffled():
    """ Assert that self.dataset is shuffled """

    atom = ATOMClassifier(X_class, y_class)
    for i in np.random.randint(0, len(X_class), 10):
        assert not atom.X.equals(X_class)


def test_percentage_data_selected():
    """ Assert that a percentage of the data is selected correctly """

    atom = ATOMClassifier(X_bin, y_bin, percentage=10, random_state=1)
    assert len(atom.X) == int(len(X_bin) * 0.10) + 1  # +1 due to rounding
    atom = ATOMClassifier(X_bin, y_bin, percentage=48, random_state=1)
    assert len(atom.y) == int(len(y_bin) * 0.48)


def test_train_test_split():
    """ Assert that the train and test split is made correctly """

    atom = ATOMClassifier(X_bin, y_bin, test_size=0.13)
    assert len(atom.train) == int(0.87*len(X_bin))


# << =============== Test update ================= >>

def test_update():
    """ Assert that the update method works as intended """

    # When dataset is changed
    atom = ATOMClassifier(X_dim4, y_dim4)
    atom.dataset.iloc[0, 2] = 20

    # Error when df is invalid
    pytest.raises(ValueError, atom.update, 'invalid')

    atom.update('dataset')
    assert atom.train.iloc[0, 2] == 20
    assert atom.X.iloc[0, 2] == 20
    assert atom.X_train.iloc[0, 2] == 20

    # When train is changed
    atom.train.iloc[1, 1] = 10
    atom.update('train')
    assert atom.dataset.iloc[1, 1] == 10
    assert atom.X.iloc[1, 1] == 10
    assert atom.X_train.iloc[1, 1] == 10

    # When y_train is changed
    atom.y_train.iloc[0] = 2
    atom.update('y_train')
    assert atom.dataset.iloc[0, -1] == 2
    assert atom.train.iloc[0, -1] == 2
    assert atom.y.iloc[0] == 2

    # When X is changed
    atom.X.iloc[0, 1] = 0.112
    atom.update('X')
    assert atom.dataset.iloc[0, 1] == 0.112
    assert atom.train.iloc[0, 1] == 0.112
    assert atom.X_train.iloc[0, 1] == 0.112


def test_index_reset():
    """ Assert that indices are reset for all data attributes """

    atom = ATOMClassifier(X_bin, y_bin)
    for attr in ['dataset', 'train', 'test', 'X', 'y',
                 'X_train', 'y_train', 'X_test', 'y_test']:
        idx = list(getattr(atom, attr).index)
        assert idx == list(range(len(getattr(atom, attr))))


def test_isPandas():
    """ Assert that data attributes are pd.DataFrames or pd.Series """

    def test(atom):
        for attr in ['dataset', 'train', 'test', 'X', 'X_train', 'X_test']:
            assert isinstance(getattr(atom, attr), pd.DataFrame)

        for attr in ['y', 'y_train', 'y_test']:
            assert isinstance(getattr(atom, attr), pd.Series)

    # Test with lists
    atom = ATOMClassifier(X_dim4, y_dim4)
    test(atom)

    # Test with np.arrays
    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    test(atom)


def test_attributes_equal_length():
    """ Assert that data attributes have the same number of rows """

    atom = ATOMClassifier(X_bin, y_bin)

    attr1, attr2 = ['X', 'X_train', 'X_test'], ['y', 'y_train', 'y_test']
    for df1, df2 in zip(attr1, attr2):
        assert len(getattr(atom, df1)) == len(getattr(atom, df2))


# << ==================== Test report ====================== >>

def test_creates_report():
    """ Assert that the report has been created and saved"""

    atom = ATOMClassifier(X_bin, y_bin)
    atom.report(rows=10)
    assert hasattr(atom, 'report')


# << ===================== Test scale ====================== >>

def test_scale():
    """ Assert that the scale method normalizes the features """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert atom.dataset.iloc[:, 1].mean() < 0.05  # Not exactly 0
    assert atom.dataset.iloc[:, 1].std() < 3


def test_already_scaled():
    """ Assert that the scale method does nothing when already scaled """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()

    atom2 = ATOMClassifier(atom.X, atom.y, random_state=1)
    X_already_scaled = atom2.X.copy()
    atom2.scale()
    assert atom2.X.equals(X_already_scaled)


# << ================ Test results ================== >>

def test_error_not_fit():
    """ Assert that an error is raised when the ATOM class is not fitted """

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(AttributeError, atom.results)


def test_error_unknown_metric():
    """ Assert that an error is raised when an unknown metric is selected """

    atom = ATOMRegressor(X_dim4, y_dim4)
    atom.pipeline(models='lgb', metric='r2', max_iter=0)
    pytest.raises(ValueError, atom.results, 'unknown')


def test_error_invalid_metric():
    """ Assert that an error is raised when an invalid metric is selected """

    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline(models='lgb', metric='r2', max_iter=0)
    pytest.raises(ValueError, atom.results, 'average_precision')


def test_al_tasks():
    """ Assert that the method works for all tasks """

    # For binary classification
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(models=['lda', 'lgb'], metric='f1', max_iter=0, bagging=3)
    atom.results()
    atom.results('jaccard')
    assert 1 == 1

    # For multiclass classification
    atom = ATOMClassifier(X_class, y_class)
    atom.pipeline(models=['pa', 'lgb'], metric='recall_macro', max_iter=0)
    atom.results()
    atom.results('f1_micro')
    assert 2 == 2

    # For regression
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline(models='lgb', metric='neg_mean_absolute_error', max_iter=0)
    atom.results()
    atom.results('neg_mean_poisson_deviance')
    assert 3 == 3
