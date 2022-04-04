# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for feature_engineering.py

"""

import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_regression

from atom.feature_engineering import (
    FeatureExtractor, FeatureGenerator, FeatureSelector,
)
from atom.utils import to_df

from .utils import (
    X10_dt, X10_str, X_bin, X_class, X_reg, X_sparse, y_bin, y_class, y_reg,
)


# Test FeatureExtractor ============================================ >>

def test_invalid_encoding_type():
    """Assert that an error is raised when encoding_type is invalid."""
    with pytest.raises(ValueError, match=r".*the encoding_type parameter.*"):
        FeatureExtractor(encoding_type="invalid").transform(X10_dt)


def test_invalid_features():
    """Assert that an error is raised when features are invalid."""
    with pytest.raises(ValueError, match=r".*an attribute of pd.Series.dt.*"):
        FeatureExtractor(features="invalid").transform(X10_dt)


def test_wrongly_converted_columns_are_ignored():
    """Assert that columns converted unsuccessfully are skipped."""
    extractor = FeatureExtractor()
    X = extractor.transform(X10_str)
    assert "feature_3" in X.columns


def test_datetime_features_are_used():
    """Assert that datetime64 features are used as is."""
    X = to_df(X10_dt.copy())
    X["feature_3"] = pd.to_datetime(X["feature_3"])

    extractor = FeatureExtractor(features="day")
    X = extractor.transform(X)
    assert "feature_3_day" in X.columns
    assert "feature_3" not in X.columns


def test_wrongly_converted_features_are_ignored():
    """Assert that wrongly converted features are ignored."""
    extractor = FeatureExtractor(features=["tz", "is_leap_year", "day"])
    X = extractor.transform(X10_dt)
    assert "feature_2_tz" not in X.columns  # Not pd.Series.dt


def test_ordinal_features():
    """Assert that ordinal features are created."""
    extractor = FeatureExtractor(features="day")
    X = extractor.transform(X10_dt)
    assert "feature_3_day" in X.columns
    assert "feature_3" not in X.columns


def test_order_features():
    """Assert that the new features are in the order provided."""
    extractor = FeatureExtractor()
    X = extractor.transform(X10_dt)
    assert X.columns.get_loc("feature_3_day") == 2
    assert X.columns.get_loc("feature_3_month") == 3
    assert X.columns.get_loc("feature_3_year") == 4


@pytest.mark.parametrize("fxs", [
    ("microsecond", "%f"),
    ("second", "%S"),
    ("hour", "%H"),
    ("weekday", "%d/%m/%Y"),
    ("day", "%d/%m/%Y"),
    ("dayofyear", "%d/%m/%Y"),
    ("month", "%d/%m/%Y"),
    ("quarter", "%d/%m/%Y"),
])
def test_all_cyclic_features(fxs):
    """Assert that all cyclic columns create two features."""
    extractor = FeatureExtractor(features=fxs[0], fmt=fxs[1], encoding_type="cyclic")
    X = extractor.transform(X10_dt)
    assert any(X.columns.str.contains(f"{fxs[0]}_cos"))
    assert X.shape[1] == 4 + 1  # 2 new and og is dropped


def test_features_are_not_dropped():
    """Assert that features are kept when drop_columns=False."""
    extractor = FeatureExtractor(drop_columns=False)
    X = extractor.transform(X10_dt)
    assert "feature_3" in X.columns


# Test FeatureGenerator ============================================ >>

def test_n_features_parameter_negative():
    """Assert that an error is raised when n_features is negative."""
    generator = FeatureGenerator(n_features=-2)
    with pytest.raises(ValueError, match=r".*the n_features parameter.*"):
        generator.fit(X_bin, y_bin)


def test_strategy_parameter():
    """Assert that the strategy parameter is either dfs or gfg."""
    generator = FeatureGenerator(strategy="invalid")
    with pytest.raises(ValueError, match=r".*strategy parameter.*"):
        generator.fit(X_bin, y_bin)


def test_operators_parameter():
    """Assert that all operators are valid."""
    generator = FeatureGenerator("gfg", n_features=None, operators=("div", "invalid"))
    with pytest.raises(ValueError, match=r".*value in the operators.*"):
        generator.fit(X_bin, y_bin)


def test_n_features_above_maximum():
    """Assert that n_features becomes maximum if more than maximum."""
    generator = FeatureGenerator(
        strategy="dfs",
        n_features=1000,
        operators="log",
        random_state=1,
    )
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 60  # 30 og + 30 log


def test_genetic_non_improving_features():
    """Assert that the code doesn't fail if there are no new improving features."""
    generator = FeatureGenerator(
        strategy="gfg",
        generations=5,
        population_size=300,
        hall_of_fame=100,
        operators="sqrt",
        random_state=1,
    )
    _ = generator.fit_transform(X_reg, y_reg)
    assert generator.genetic_features is None


def test_attribute_genetic_features():
    """Assert that the genetic_features attribute is created."""
    generator = FeatureGenerator(
        strategy="gfg",
        generations=3,
        population_size=400,
        random_state=1,
    )
    _ = generator.fit_transform(X_bin, y_bin)
    assert not generator.genetic_features.empty


def test_updated_dataset():
    """Assert that the feature set contains the new features."""
    generator = FeatureGenerator(
        strategy="gfg",
        n_features=1,
        generations=4,
        population_size=1000,
        random_state=1,
    )
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] == X_bin.shape[1] + 1

    generator = FeatureGenerator(strategy="dfs", n_features=None, random_state=1)
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] > X_bin.shape[1]


def test_default_feature_names():
    """Assert that the new features get correct default names."""
    X = X_bin.copy()
    X["feature_32"] = range(len(X))

    generator = FeatureGenerator(strategy="gfg", n_features=2, random_state=1)
    X = generator.fit_transform(X, y_bin)
    assert "feature_31" not in X
    assert "feature_34" in X and "feature_33" in X


# Test FeatureSelector ============================================= >>

def test_unknown_strategy_parameter():
    """Assert that an error is raised when strategy is unknown."""
    selector = FeatureSelector(strategy="invalid")
    with pytest.raises(ValueError, match=r".*the strategy parameter.*"):
        selector.fit(X_reg, y_reg)


def test_solver_parameter_empty():
    """Assert that an error is raised when solver is None."""
    selector = FeatureSelector(strategy="sfm", solver=None)
    with pytest.raises(ValueError, match=r".*can't be None.*"):
        selector.fit(X_reg, y_reg)


def test_goal_attribute():
    """Assert that the goal is deduced from the model's name."""
    # For classification tasks
    selector = FeatureSelector(strategy="sfm", solver="LGB_class")
    selector.fit(X_bin, y_bin)
    assert selector.goal == "class"

    # For regression tasks
    selector = FeatureSelector(strategy="sfm", solver="LGB_reg")
    selector.fit(X_reg, y_reg)
    assert selector.goal == "reg"


def test_solver_parameter_invalid_value():
    """Assert that an error is raised when solver is unknown."""
    selector = FeatureSelector(strategy="RFE", solver="invalid")
    with pytest.raises(ValueError, match=r".*Unknown model.*"):
        selector.fit(X_reg, y_reg)


def test_kwargs_but_no_strategy():
    """Assert that an error is raised when kwargs are defined and strategy=None."""
    selector = FeatureSelector(strategy=None, cv=2)
    with pytest.raises(ValueError, match=r".*Keyword arguments.*"):
        selector.fit(X_reg, y_reg)


def test_n_features_parameter():
    """Assert that an error is raised when n_features is invalid."""
    selector = FeatureSelector(strategy="sfm", solver="XGB_reg", n_features=0)
    with pytest.raises(ValueError, match=r".*the n_features parameter.*"):
        selector.fit(X_reg, y_reg)


def test_max_frac_repeated_parameter():
    """Assert that an error is raised when max_frac_repeated is invalid."""
    selector = FeatureSelector(strategy=None, max_frac_repeated=1.1)
    with pytest.raises(ValueError, match=r".*the max_frac_repeated parameter.*"):
        selector.fit(X_reg, y_reg)


def test_max_correlation_parameter():
    """Assert that an error is raised when max_correlation is invalid."""
    selector = FeatureSelector(strategy=None, max_correlation=-0.2)
    with pytest.raises(ValueError, match=r".*the max_correlation parameter.*"):
        selector.fit(X_reg, y_reg)


def test_error_y_is_None():
    """Assert that an error is raised when y is None for some strategies."""
    selector = FeatureSelector(strategy="univariate", solver=f_regression, n_features=9)
    with pytest.raises(ValueError, match=r".*the y parameter.*"):
        selector.fit(X_reg)


def test_remove_low_variance():
    """Assert that the remove_low_variance function works as intended."""
    X = X_bin.copy()
    X["invalid"] = 3  # Add column with minimum variance
    selector = FeatureSelector(max_frac_repeated=1.0)
    X = selector.fit_transform(X)
    assert X.shape[1] == X_bin.shape[1]


def test_collinear_attribute():
    """Assert that the collinear attribute is created."""
    selector = FeatureSelector(max_correlation=0.6)
    assert hasattr(selector, "collinear")


def test_remove_collinear():
    """Assert that the remove_collinear function works as intended."""
    selector = FeatureSelector(max_correlation=0.9)
    X = selector.fit_transform(X_bin)
    assert X.shape[1] == 20  # Originally 30


def test_solver_parameter_empty_univariate():
    """Assert that an error is raised when solver is None for univariate."""
    selector = FeatureSelector(strategy="univariate")
    with pytest.raises(ValueError, match=r".*can't be None.*"):
        selector.fit(X_reg, y_reg)


def test_raise_unknown_solver_univariate():
    """Assert that an error is raised when the solver is unknown."""
    selector = FeatureSelector(strategy="univariate", solver="invalid")
    with pytest.raises(ValueError, match=r".*the solver parameter.*"):
        selector.fit(X_reg, y_reg)


def test_univariate_strategy_custom_solver():
    """Assert that the univariate strategy works for a custom solver."""
    selector = FeatureSelector("univariate", solver=f_regression, n_features=9)
    X = selector.fit_transform(X_reg, y_reg)
    assert X.shape[1] == 9
    assert set(selector.feature_importance) == set(X.columns)


def test_pca_strategy():
    """Assert that the pca strategy works as intended."""
    selector = FeatureSelector(strategy="pca", n_features=0.7)
    X = selector.fit_transform(X_bin)
    assert X.shape[1] == 21
    assert selector.pca.get_params()["svd_solver"] == "auto"


def test_pca_components():
    """Assert that the pca strategy creates components instead of features."""
    selector = FeatureSelector(strategy="pca", solver="arpack", n_features=5)
    X = selector.fit_transform(X_bin)
    assert selector.pca.svd_solver == "arpack"
    assert "component_1" in X.columns


def test_pca_sparse_data():
    """Assert that the pca strategy uses TruncatedSVD for sparse data."""
    selector = FeatureSelector(strategy="pca", n_features=2)
    selector.fit(X_sparse)
    assert selector.pca.__class__.__name__ == "TruncatedSVD"
    assert selector.pca.get_params()["algorithm"] == "randomized"


def test_sfm_prefit_invalid_estimator():
    """Assert that an error is raised for an invalid estimator in sfm."""
    selector = FeatureSelector(
        strategy="sfm",
        solver=ExtraTreesClassifier(random_state=1).fit(X_class, y_class),
        n_features=8,
        random_state=1,
    )
    with pytest.raises(ValueError, match=r".*different columns than X.*"):
        selector.fit(X_bin, y_bin)


def test_sfm_strategy_not_threshold():
    """Assert that if threshold is not specified, sfm selects n_features features."""
    selector = FeatureSelector(
        strategy="sfm",
        solver=ExtraTreesClassifier(random_state=1),
        n_features=16,
        random_state=1,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 16


def test_sfm_invalid_solver():
    """Assert that an error is raised when solver is invalid."""
    selector = FeatureSelector(strategy="sfm", solver="invalid", n_features=5)
    with pytest.raises(ValueError, match=r".*Unknown model.*"):
        selector.fit_transform(X_bin, y_bin)


def test_sfm_strategy_fitted_solver():
    """Assert that the sfm strategy works when the solver is already fitted."""
    selector = FeatureSelector(
        strategy="sfm",
        solver=ExtraTreesClassifier(random_state=1).fit(X_bin, y_bin),
        n_features=7,
        random_state=1,
    )
    X = selector.fit_transform(X_bin)
    assert X.shape[1] == 7
    assert set(selector.feature_importance) == set(X.columns)


def test_sfm_strategy_not_fitted_solver():
    """Assert that the sfm strategy works when the solver is not fitted."""
    selector = FeatureSelector(
        strategy="sfm", solver=ExtraTreesClassifier(random_state=1), n_features=5
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 5
    assert set(selector.feature_importance) == set(X.columns)


def test_sfs_strategy():
    """Assert that the sfs strategy works."""
    selector = FeatureSelector(
        strategy="sfs",
        solver="RF_reg",
        n_features=6,
        cv=3,
        random_state=1,
    )
    X = selector.fit_transform(X_reg, y_reg)
    assert X.shape[1] == 6


def test_RFE_strategy():
    """Assert that the RFE strategy works as intended."""
    selector = FeatureSelector(
        strategy="RFE",
        solver=ExtraTreesClassifier(random_state=1),
        n_features=13,
        random_state=1,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 13
    assert set(selector.feature_importance) == set(X.columns)


def test_rfecv_strategy_before_pipeline_classification():
    """Assert that the rfecv strategy works before a fitted pipeline."""
    selector = FeatureSelector(
        strategy="rfecv",
        solver="Tree_class",
        n_features=None,
        random_state=1,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 4
    assert set(selector.feature_importance) == set(X.columns)


def test_rfecv_strategy_before_pipeline_regression():
    """Assert that the rfecv strategy works before a fitted pipeline."""
    selector = FeatureSelector("rfecv", solver="RF_reg", n_features=16, random_state=1)
    X = selector.fit_transform(X_reg, y_reg)
    assert X.shape[1] == 10
    assert set(selector.feature_importance) == set(X.columns)


def test_kwargs_parameter_threshold():
    """Assert that the kwargs parameter works as intended (add threshold)."""
    selector = FeatureSelector(
        strategy="sfm",
        solver=ExtraTreesClassifier(random_state=1),
        n_features=21,
        threshold="mean",
        random_state=1,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 10


def test_kwargs_parameter_tol():
    """Assert that the kwargs parameter works as intended (add tol)."""
    selector = FeatureSelector(
        strategy="pca",
        solver="arpack",
        tol=0.001,
        n_features=12,
        max_frac_repeated=None,
        max_correlation=None,
        random_state=1,
    )
    X = selector.fit_transform(X_bin)
    assert X.shape[1] == 12


@pytest.mark.parametrize("strategy", ["sfs", "rfecv", "pso"])
def test_kwargs_parameter_scoring(strategy):
    """Assert that the kwargs parameter works as intended (add scoring acronym)."""
    selector = FeatureSelector(
        strategy=strategy,
        solver="tree_class",
        scoring="auc",
        n_features=12,
        max_frac_repeated=None,
        max_correlation=None,
        random_state=1,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] < X_bin.shape[1]


def test_advanced_provided_validation_sets():
    """Assert that custom validation sets can be provided."""
    selector = FeatureSelector(
        strategy="pso",
        solver="tree_class",
        X_valid=X_bin.iloc[:20, :],
        y_valid=y_bin.iloc[:20],
        n_iteration=2,
        population_size=2,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] < X_bin.shape[1]


def test_advanced_missing_y_valid():
    """Assert that an error is raised when y_valid is missing."""
    selector = FeatureSelector("pso", solver="tree_class", X_valid=X_bin)
    with pytest.raises(ValueError, match=r".*y_valid parameter.*"):
        selector.fit(X_bin, y_bin)


def test_advanced_custom_scoring():
    """Assert that scoring can be specified by the user."""
    selector = FeatureSelector(
        strategy="pso",
        solver="tree_class",
        n_iteration=2,
        population_size=2,
        scoring="auc",
    )
    selector = selector.fit(X_bin, y_bin)
    assert selector.pso.kwargs["scoring"].name == "roc_auc"


def test_advanced_binary_classification_scoring():
    """Assert that scoring is set for binary classification tasks."""
    selector = FeatureSelector(
        strategy="pso",
        solver="tree_class",
        n_iteration=2,
        population_size=2,
    )
    selector = selector.fit(X_bin, y_bin)
    assert selector.pso.kwargs["scoring"].name == "f1"


def test_advanced_multiclass_classification_scoring():
    """Assert that scoring is set for multiclass classification tasks."""
    selector = FeatureSelector(
        strategy="pso",
        solver="tree_class",
        n_iteration=2,
        population_size=2,
    )
    selector = selector.fit(X_class, y_class)
    assert selector.pso.kwargs["scoring"].name == "f1_weighted"


def test_advanced_regression_scoring():
    """Assert that scoring is set for regression tasks."""
    selector = FeatureSelector(
        strategy="hho",
        solver="tree_reg",
        n_iteration=2,
        population_size=2,
    )
    selector = selector.fit(X_reg, y_reg)
    assert selector.hho.kwargs["scoring"].name == "r2"


def test_advanced_custom_objective_function():
    """Assert that a custom objective function can be used."""
    selector = FeatureSelector(
        strategy="gwo",
        solver="tree_class",
        objective_function=lambda *args: 1,
        n_iteration=2,
        population_size=2,
    )
    selector = selector.fit(X_bin, y_bin)
    assert selector.gwo.objective_function.__name__ == "<lambda>"
