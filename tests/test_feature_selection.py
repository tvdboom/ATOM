# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the feature_selection method of the ATOM class.

'''

# Import packages
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer, load_boston
from atom import ATOMClassifier


# << ====================== Functions ====================== >>

def load_df(dataset):
    ''' Load dataset as pd.DataFrame '''

    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    data = pd.DataFrame(data, columns=columns)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y


# << ======================= Tests ========================= >>

# << =================== Test parameters =================== >>

def test_strategy_parameter():
    ''' Assert that the strategy parameter is set correctly '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    pytest.raises(TypeError, atom.feature_selection, strategy=True)
    pytest.raises(ValueError, atom.feature_selection, strategy='test')


def test_max_features_parameter():
    ''' Assert that the max_features parameter is set correctly '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    pytest.raises(TypeError, atom.feature_selection, max_features='test')
    pytest.raises(ValueError, atom.feature_selection, max_features=0)


def test_threshold_parameter():
    ''' Assert that the threshold parameter is set correctly '''

    # When wrong type
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    pytest.raises(TypeError, atom.feature_selection, threshold=[2, 2])

    # When wrong string value
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    pytest.raises(ValueError,
                  atom.feature_selection,
                  strategy='sfm',
                  solver=RandomForestClassifier(),
                  threshold='test')


def test_min_variance_frac_parameter():
    ''' Assert that the min_variance_frac parameter is set correctly '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    pytest.raises(TypeError, atom.feature_selection, min_variance_frac='test')
    pytest.raises(ValueError, atom.feature_selection, min_variance_frac=1.1)


def test_max_correlation_parameter():
    ''' Assert that the max_correlation parameter is set correctly '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    pytest.raises(TypeError, atom.feature_selection, max_correlation='test')
    pytest.raises(ValueError, atom.feature_selection, max_correlation=-0.2)


# << ==================== Test functions =================== >>

def test_remove_low_variance():
    ''' Assert that the remove_low_variance function works as intended '''

    X, y = load_df(load_breast_cancer())
    X['test_column'] = 3  # Add column with minimum variance
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(min_variance_frac=1., max_correlation=None)
    assert len(atom.X.columns) + 1 == len(X.columns)


def test_collinear_attribute():
    ''' Assert that the collinear attribute is created '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(max_correlation=0.6)
    assert hasattr(atom, 'collinear')


def test_remove_collinear():
    ''' Assert that the remove_collinear function works as intended '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(max_correlation=0.9)
    assert len(atom.X.columns) == 20  # Originally 30


# << ==================== Test strategies ================== >>

def test_raise_unknown_solver():
    ''' Assert that an error is raised when the solver is unknown '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    pytest.raises(ValueError,
                  atom.feature_selection,
                  strategy='univariate',
                  solver='test')


def test_univariate_attribute():
    ''' Assert that the univariate attribute is created '''

    # For classification tasks
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='univariate', solver='mutual_info_classif')
    assert hasattr(atom, 'univariate')

    # For regression tasks (different solver)
    X, y = load_df(load_boston())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='univariate')
    assert hasattr(atom, 'univariate')


def test_univariate_strategy():
    ''' Assert that the univariate strategy works as intended '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='univariate',
                           max_features=9,
                           max_correlation=None)
    assert len(atom.X.columns) == 9  # Assert number of features


def test_PCA_attribute():
    ''' Assert that the PCA attribute is created '''

    # For classification tasks
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='pca', max_features=2, solver=None)
    assert hasattr(atom, 'PCA')

    # For regression tasks (different solver)
    X, y = load_df(load_boston())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='PCA', solver='arpack', max_features=2)
    assert hasattr(atom, 'PCA')


def test_PCA_strategy_normalization():
    ''' Assert that the PCA strategy normalizes the features '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='pca', max_correlation=None)
    assert atom.dataset.iloc[:, 1].mean() < 0.05  # Not exactly 0
    assert atom.dataset.iloc[:, 1].std() < 3


def test_PCA_strategy():
    ''' Assert that the PCA strategy works as intended '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='pca',
                           max_features=12,
                           max_correlation=None)
    assert len(atom.X.columns) == 12  # Assert number of features


def test_PCA_components():
    ''' Assert that the PCA strategy creates components instead of features '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='pca', max_correlation=None)
    assert 'Component 0' in atom.X.columns


def test_SFM_attribute():
    ''' Assert that the SFM attribute is created '''

    # For classification tasks
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='sfm', solver=RandomForestClassifier())
    assert hasattr(atom, 'SFM')

    # For regression tasks
    X, y = load_df(load_boston())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='SFM', solver=RandomForestRegressor())
    assert hasattr(atom, 'SFM')


def test_SFM_strategy():
    ''' Assert that the SFM strategy works as intended '''

    # For fitted solver
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    rf = RandomForestClassifier().fit(atom.X_test, atom.y_test)
    atom.feature_selection(strategy='sfm',
                           solver=rf,
                           max_features=7,
                           max_correlation=None)
    assert len(atom.X.columns) == 7  # Assert number of features

    # For unfitted solver
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='sfm',
                           solver=RandomForestClassifier(),
                           max_features=5,
                           max_correlation=None)
    assert len(atom.X.columns) == 5  # Assert number of features


def test_SFM_threshold():
    ''' Assert that the threshold parameter works as intended '''

    # For numerical value
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='sfm',
                           solver=RandomForestClassifier(),
                           max_features=11,
                           threshold=0.1,
                           max_correlation=None)
    assert len(atom.X.columns) == 4  # Assert number of features

    # For string value
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='sfm',
                           solver=RandomForestClassifier(),
                           max_features=19,
                           threshold='mean',
                           max_correlation=None)
    assert len(atom.X.columns) == 10  # Assert number of features


def test_RFE_attribute():
    ''' Assert that the RFE attribute is created '''

    # For classification tasks
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='rfe', solver=RandomForestClassifier())
    assert hasattr(atom, 'RFE')

    # For regression tasks
    X, y = load_df(load_boston())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='rfe', solver=RandomForestRegressor())
    assert hasattr(atom, 'RFE')


def test_RFE_strategy():
    ''' Assert that the RFE strategy works as intended '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.feature_selection(strategy='sfm',
                           solver=RandomForestClassifier(),
                           max_features=13,
                           max_correlation=None)
    assert len(atom.X.columns) == 13  # Assert number of features
