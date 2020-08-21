# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for feature_engineering.py

"""

# Standard packages
import pytest
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_regression

# Own modules
from atom.feature_engineering import FeatureGenerator, FeatureSelector
from atom.utils import check_scaling
from .utils import X_bin, y_bin, X_reg, y_reg


# Test FeatureGenerator ===================================================== >>

def test_n_features_parameter_negative():
    """Assert that an error is raised when n_features is negative."""
    with pytest.raises(ValueError, match=r".*should be >0.*"):
        FeatureGenerator(n_features=-2)


def test_population_parameter():
    """Assert that an error is raised when population is invalid."""
    pytest.raises(ValueError, FeatureGenerator, 'genetic', population=30)


def test_generations_parameter():
    """Assert that an error is raised when generations is invalid."""
    pytest.raises(ValueError, FeatureGenerator, 'genetic', generations=0)


def test_n_features_parameter_not_one_percent():
    """Assert that the n_features parameter is within 1% of population."""
    with pytest.raises(ValueError, match=r".*should be <1%.*"):
        FeatureGenerator('genetic', n_features=23, population=200)


def test_strategy_parameter():
    """Assert that the strategy parameter is either 'dfs' or 'genetic'."""
    with pytest.raises(ValueError, match=r".*should be either 'dfs'.*"):
        FeatureGenerator('unknown', n_features=None)


def test_operators_parameter():
    """Assert that all operators are valid."""
    with pytest.raises(ValueError, match=r".*value in the operators.*"):
        FeatureGenerator('gfg', n_features=None, operators=('div', 'invalid'))


def test_n_features_above_maximum():
    """Assert that n_features becomes maximum if more than maximum for 'DFS'."""
    generator = FeatureGenerator(n_features=1000, operators='log', random_state=1)
    _ = generator.fit_transform(X_bin, y_bin)
    assert generator.n_features == 30


def test_attribute_genetic_features():
    """Assert that the genetic_features attribute is created."""
    generator = FeatureGenerator(strategy='gfg',
                                 generations=3,
                                 population=200,
                                 random_state=1)
    _ = generator.fit_transform(X_bin, y_bin)
    assert isinstance(generator.genetic_features, pd.DataFrame)


def test_genetic_maximum_features():
    """Assert that the features are 1% of the population for n_features=None."""
    generator = FeatureGenerator(strategy='gfg',
                                 n_features=None,
                                 generations=4,
                                 population=400,
                                 random_state=1)
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] == X_bin.shape[1] + 4


def test_updated_dataset():
    """Assert that the feature set contains the new features."""
    generator = FeatureGenerator(strategy='gfg',
                                 n_features=1,
                                 generations=4,
                                 population=1000,
                                 random_state=1)
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] == X_bin.shape[1] + 1

    generator = FeatureGenerator(strategy='dfs', n_features=None, random_state=1)
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] > X_bin.shape[1]


# Test FeatureSelector ====================================================== >>

def test_unknown_strategy_parameter():
    """Assert that an error is raised when strategy is unknown."""
    pytest.raises(ValueError, FeatureSelector, strategy='test')


def test_solver_parameter_empty_univariate():
    """Assert that an error is raised when solver is None for univariate."""
    pytest.raises(ValueError,  FeatureSelector, strategy='univariate')


def test_raise_unknown_solver_univariate():
    """Assert that an error is raised when the solver is unknown."""
    pytest.raises(ValueError, FeatureSelector, 'univariate', solver='test')


def test_solver_auto_PCA():
    """Assert that the solver is set to 'auto' when None."""
    fs = FeatureSelector(strategy='PCA', solver=None)
    assert fs.solver == 'auto'


def test_solver_parameter_empty_SFM():
    """Assert that an error is raised when solver is None for SFM strategy."""
    pytest.raises(ValueError,  FeatureSelector, strategy='SFM')


def test_goal_attribute():
    """Assert that the goal is deduced from the model's name."""
    # For classification tasks
    fs = FeatureSelector(strategy='SFM', solver='LGB_class')
    assert fs.goal == 'classification'

    # For regression tasks
    fs = FeatureSelector(strategy='SFM', solver='LGB_reg')
    assert fs.goal == 'regression'


def test_solver_parameter_invalid_value():
    """Assert that an error is raised when solver is unknown."""
    pytest.raises(ValueError, FeatureSelector, strategy='rfe', solver='test')


def test_n_features_parameter():
    """Assert that an error is raised when n_features is invalid."""
    pytest.raises(ValueError, FeatureSelector, n_features=0)


def test_max_frac_repeated_parameter():
    """Assert that an error is raised when max_frac_repeated is invalid."""
    pytest.raises(ValueError, FeatureSelector, max_frac_repeated=1.1)


def test_max_correlation_parameter():
    """Assert that an error is raised when max_correlation is invalid."""
    pytest.raises(ValueError, FeatureSelector, max_correlation=-0.2)


def test_error_y_is_None():
    """Assert that an error is raised when y is None for some strategies."""
    fs = FeatureSelector('univariate', solver=f_regression, n_features=9)
    pytest.raises(ValueError, fs.fit, X_reg)


def test_remove_low_variance():
    """Assert that the remove_low_variance function works as intended."""
    X = X_bin.copy()
    X['test'] = 3  # Add column with minimum variance
    fs = FeatureSelector(max_frac_repeated=1.)
    X = fs.fit_transform(X)
    assert X.shape[1] == X_bin.shape[1]


def test_collinear_attribute():
    """Assert that the collinear attribute is created."""
    fs = FeatureSelector(max_correlation=0.6)
    assert hasattr(fs, 'collinear')


def test_remove_collinear():
    """Assert that the remove_collinear function works as intended."""
    fs = FeatureSelector(max_correlation=0.9)
    X = fs.fit_transform(X_bin)
    assert X.shape[1] == 20  # Originally 30


def test_univariate_strategy_custom_solver():
    """Assert that the univariate strategy works for a custom solver."""
    fs = FeatureSelector('univariate', solver=f_regression, n_features=9)
    X = fs.fit_transform(X_reg, y_reg)
    assert X.shape[1] == 9


def test_PCA_strategy_normalization():
    """Assert that the PCA strategy normalizes the features."""
    fs = FeatureSelector(strategy='PCA')
    X = fs.fit_transform(X_bin, y_bin)
    assert check_scaling(X)


def test_PCA_strategy():
    """Assert that the PCA strategy works as intended."""
    fs = FeatureSelector(strategy='PCA', n_features=0.7)
    X = fs.fit_transform(X_bin)
    assert X.shape[1] == 21


def test_PCA_components():
    """Assert that the PCA strategy creates components instead of features."""
    fs = FeatureSelector(strategy='PCA')
    X = fs.fit_transform(X_bin)
    assert 'Component 0' in X.columns


def test_SFM_strategy_not_threshold():
    """Assert that if threshold is not specified, SFM selects n_features features."""
    fs = FeatureSelector(strategy='SFM',
                         solver=ExtraTreesClassifier(random_state=1),
                         n_features=16)
    X = fs.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 16


def test_SFM_strategy_fitted_solver():
    """Assert that the SFM strategy works when the solver is already fitted."""
    rf = ExtraTreesClassifier().fit(X_bin, y_bin)
    fs = FeatureSelector('SFM', solver=rf, n_features=7)
    X = fs.fit_transform(X_bin)
    assert X.shape[1] == 7


def test_SFM_strategy_not_fitted_solver():
    """Assert that the SFM strategy works when the solver is not fitted."""
    fs = FeatureSelector('SFM', solver=ExtraTreesClassifier(), n_features=5)
    X = fs.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 5


def test_RFE_strategy():
    """Assert that the RFE strategy works as intended."""
    fs = FeatureSelector('RFE', solver=ExtraTreesClassifier(), n_features=13)
    X = fs.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 13


def test_RFECV_strategy_before_pipeline_classification():
    """Assert that the RFECV strategy works before a fitted pipeline."""
    fs = FeatureSelector('RFECV', solver='LGB_class', n_features=None)
    X = fs.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 20


def test_RFECV_strategy_before_pipeline_regression():
    """Assert that the RFECV strategy works before a fitted pipeline."""
    fs = FeatureSelector('RFECV', solver='LGB_reg', n_features=16)
    X = fs.fit_transform(X_reg, y_reg)
    assert X.shape[1] == 10


def test_kwargs_parameter_threshold():
    """Assert that the kwargs parameter works as intended (add threshold)."""
    fs = FeatureSelector(strategy='SFM',
                         solver=ExtraTreesClassifier(random_state=1),
                         n_features=21,
                         threshold='mean')
    X = fs.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 10


def test_kwargs_parameter_tol():
    """Assert that the kwargs parameter works as intended (add tol)."""
    fs = FeatureSelector(strategy='PCA',
                         solver='arpack',
                         tol=0.001,
                         n_features=12)
    X = fs.fit_transform(X_bin)
    assert X.shape[1] == 12


def test_kwargs_parameter_scoring():
    """Assert that the kwargs parameter works as intended (add scoring acronym)."""
    fs = FeatureSelector(strategy='RFECV',
                         solver='lgb_class',
                         scoring='auc',
                         n_features=12)
    X = fs.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 24
