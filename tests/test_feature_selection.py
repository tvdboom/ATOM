# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the feature_selection method of the ATOM class.

"""

# Import packages
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Variables ====================== >>

X_bin, y_bin = load_breast_cancer(return_X_y=True)
X_reg, y_reg = load_boston(return_X_y=True)


# << ======================= Tests ========================= >>

# << =================== Test parameters =================== >>

def test_strategy_parameter():
    """ Assert that the strategy parameter is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(ValueError, atom.feature_selection, strategy='test')


def test_solver_parameter():
    """ Assert that solver raises an error """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # When left empty
    pytest.raises(ValueError,  atom.feature_selection, strategy='sfm')

    # When invalid value
    pytest.raises(ValueError,
                  atom.feature_selection,
                  strategy='rfe',
                  solver='invalid')


def test_n_features_parameter():
    """ Assert that the n_features parameter is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(ValueError, atom.feature_selection, n_features=0)


def test_max_frac_repeated_parameter():
    """ Assert that the max_frac_repeated parameter is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(ValueError, atom.feature_selection, max_frac_repeated=1.1)


def test_max_correlation_parameter():
    """ Assert that the max_correlation parameter is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(ValueError, atom.feature_selection, max_correlation=-0.2)


# << ==================== Test functions =================== >>

def test_remove_low_variance():
    """ Assert that the remove_low_variance function works as intended """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X['test'] = 3  # Add column with minimum variance
    atom.update('X')
    atom.feature_selection(max_frac_repeated=0.98, max_correlation=None)
    assert len(atom.X.columns) == X_bin.shape[1]


def test_collinear_attribute():
    """ Assert that the collinear attribute is created """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(max_correlation=0.6)
    assert hasattr(atom, 'collinear')


def test_remove_collinear():
    """ Assert that the remove_collinear function works as intended """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(max_correlation=0.9)
    assert len(atom.X.columns) == 20  # Originally 30


# << ==================== Test strategies ================== >>

def test_raise_unknown_solver():
    """ Assert that an error is raised when the solver is unknown """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError,
                  atom.feature_selection,
                  strategy='univariate',
                  solver='test')


def test_winner_model_solver():
    """ Assert that the solver uses the winner model if it exists """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['lr', 'bnb'])
    atom.feature_selection('SFM', solver=None, n_features=12)
    assert atom.feature_selector.solver is atom.winner.best_model_fit


def test_univariate_strategy():
    """ Assert that the univariate strategy works as intended """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='univariate',
                           n_features=9,
                           max_correlation=None)
    assert len(atom.X.columns) == 9  # Assert number of features


def test_PCA_strategy_normalization():
    """ Assert that the PCA strategy normalizes the features """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='pca', max_correlation=None)
    assert atom.dataset.iloc[:, 1].mean() < 0.05  # Not exactly 0
    assert atom.dataset.iloc[:, 1].std() < 3


def test_PCA_strategy():
    """ Assert that the PCA strategy works as intended """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='pca',
                           n_features=12,
                           max_correlation=None)
    assert len(atom.X.columns) == 12  # Assert number of features


def test_PCA_components():
    """ Assert that the PCA strategy creates components instead of features """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='pca', max_correlation=None)
    assert 'Component 0' in atom.X.columns


def test_SFM_strategy():
    """ Assert that the SFM strategy works as intended """

    # For fitted solver
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    rf = RandomForestClassifier().fit(atom.X_test, atom.y_test)
    atom.feature_selection(strategy='sfm',
                           solver=rf,
                           n_features=7,
                           max_correlation=None)
    assert len(atom.X.columns) == 7  # Assert number of features

    # For unfitted solver
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='sfm',
                           solver=RandomForestClassifier(),
                           n_features=5,
                           max_correlation=None)
    assert len(atom.X.columns) == 5  # Assert number of features


def test_RFE_strategy():
    """ Assert that the RFE strategy works as intended """

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='rfe',
                           solver=RandomForestClassifier(),
                           n_features=13,
                           max_correlation=None)
    assert len(atom.X.columns) == 13  # Assert number of features


def test_RFECV_strategy():
    """ Assert that the RFECV strategy works as intended """

    # Before pipeline
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='rfecv',
                           solver='lgb',
                           n_features=16,
                           max_correlation=None)
    assert len(atom.X.columns) == 23

    # After pipeline (uses selected metric)
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline('lgb', metric='accuracy')
    atom.feature_selection(strategy='rfecv',
                           solver='lgb',
                           n_features=16,
                           max_correlation=None)
    params = atom.feature_selector.RFECV.get_params()
    assert params['scoring'].name == 'accuracy'


def test_kwargs_parameter():
    """ Assert that the kwargs parameter works as intended """

    # Add the threshold parameter to the SFM strategy
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='sfm',
                           solver=RandomForestClassifier(),
                           n_features=11,
                           max_correlation=None,
                           threshold=0.1,)
    assert len(atom.X.columns) == 4  # Assert number of features

    # Add tol parameter to PCA strategy
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='pca',
                           solver='arpack',
                           tol=0.001,
                           n_features=12,
                           max_correlation=None)
    assert len(atom.X.columns) == 12
