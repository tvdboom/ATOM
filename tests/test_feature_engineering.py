# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for feature_engineering.py

"""

# Standard packages
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_regression

# Own modules
from atom.feature_engineering import (
    FeatureExtractor,
    FeatureGenerator,
    FeatureSelector,
)
from atom.utils import to_df
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg, X10_str, X10_dt


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
    assert "Feature 3" in X.columns


def test_datetime_features_are_used():
    """Assert that datetime64 features are used as is."""
    X = to_df(X10_dt.copy())
    X["Feature 3"] = pd.to_datetime(X["Feature 3"])

    extractor = FeatureExtractor(features="day")
    X = extractor.transform(X)
    assert "Feature 3_day" in X.columns
    assert "Feature 3" not in X.columns


def test_wrongly_converted_features_are_ignored():
    """Assert that wrongly converted features are ignored."""
    extractor = FeatureExtractor(features=["tz", "is_leap_year", "day"])
    X = extractor.transform(X10_dt)
    assert "Feature 2_tz" not in X.columns  # Not pd.Series.dt


def test_ordinal_features():
    """Assert that ordinal features are created."""
    extractor = FeatureExtractor(features="day")
    X = extractor.transform(X10_dt)
    assert "Feature 3_day" in X.columns
    assert "Feature 3" not in X.columns


def test_order_features():
    """Assert that the new features are in the order provided."""
    extractor = FeatureExtractor()
    X = extractor.transform(X10_dt)
    assert X.columns.get_loc("Feature 3_day") == 2
    assert X.columns.get_loc("Feature 3_month") == 3
    assert X.columns.get_loc("Feature 3_year") == 4


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
    assert "Feature 3" in X.columns


# Test FeatureGenerator ============================================ >>

def test_n_features_parameter_negative():
    """Assert that an error is raised when n_features is negative."""
    generator = FeatureGenerator(n_features=-2)
    with pytest.raises(ValueError, match=r".*should be >0.*"):
        generator.fit(X_bin, y_bin)


def test_population_parameter():
    """Assert that an error is raised when population is invalid."""
    generator = FeatureGenerator(strategy="gfg", population=30)
    pytest.raises(ValueError, generator.fit, X_reg, y_reg)


def test_generations_parameter():
    """Assert that an error is raised when generations is invalid."""
    generator = FeatureGenerator(strategy="gfg", generations=0)
    pytest.raises(ValueError, generator.fit, X_bin, y_bin)


def test_n_features_parameter_not_one_percent():
    """Assert that the n_features parameter is within 1% of population."""
    generator = FeatureGenerator(strategy="gfg", n_features=23, population=200)
    with pytest.raises(ValueError, match=r".*should be <1%.*"):
        generator.fit(X_bin, y_bin)


def test_strategy_parameter():
    """Assert that the strategy parameter is either "DFS", "GFG" or "genetic"."""
    generator = FeatureGenerator(strategy="invalid")
    with pytest.raises(ValueError, match=r".*should be either 'dfs'.*"):
        generator.fit(X_bin, y_bin)


def test_operators_parameter():
    """Assert that all operators are valid."""
    generator = FeatureGenerator("GFG", n_features=None, operators=("div", "invalid"))
    with pytest.raises(ValueError, match=r".*value in the operators.*"):
        generator.fit(X_bin, y_bin)


def test_n_features_above_maximum():
    """Assert that n_features becomes maximum if more than maximum for "DFS"."""
    generator = FeatureGenerator(n_features=1000, operators="log", random_state=1)
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 60  # 30 og + 30 log


def test_genetic_non_improving_features():
    """Assert that the code doesn't fail if there are no new improving features."""
    generator = FeatureGenerator(
        strategy="gfg",
        generations=5,
        population=300,
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
        population=200,
        random_state=1,
    )
    _ = generator.fit_transform(X_bin, y_bin)
    assert not generator.genetic_features.empty


def test_genetic_maximum_features():
    """Assert that the features are 1% of the population for n_features=None."""
    generator = FeatureGenerator(
        strategy="gfg",
        n_features=None,
        generations=4,
        population=400,
        random_state=1,
    )
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] == X_bin.shape[1] + 4


def test_updated_dataset():
    """Assert that the feature set contains the new features."""
    generator = FeatureGenerator(
        strategy="gfg",
        n_features=1,
        generations=4,
        population=1000,
        random_state=1,
    )
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] == X_bin.shape[1] + 1

    generator = FeatureGenerator(strategy="dfs", n_features=None, random_state=1)
    X = generator.fit_transform(X_bin, y_bin)
    assert X.shape[1] > X_bin.shape[1]


# Test FeatureSelector ============================================= >>

def test_unknown_strategy_parameter():
    """Assert that an error is raised when strategy is unknown."""
    selector = FeatureSelector(strategy="invalid")
    pytest.raises(ValueError, selector.fit, X_reg, y_reg)


def test_solver_parameter_empty_univariate():
    """Assert that an error is raised when solver is None for univariate."""
    selector = FeatureSelector(strategy="univariate")
    pytest.raises(ValueError, selector.fit, X_reg, y_reg)


def test_raise_unknown_solver_univariate():
    """Assert that an error is raised when the solver is unknown."""
    selector = FeatureSelector(strategy="univariate", solver="invalid")
    pytest.raises(ValueError, selector.fit, X_reg, y_reg)


def test_solver_auto_PCA():
    """Assert that the solver is set to "auto" when None."""
    selector = FeatureSelector(strategy="PCA", solver=None)
    selector.fit(X_bin, y_bin)
    assert selector._solver == "auto"


def test_solver_parameter_empty_SFM():
    """Assert that an error is raised when solver is None for SFM strategy."""
    selector = FeatureSelector(strategy="SFM", solver=None)
    pytest.raises(ValueError, selector.fit, X_reg, y_reg)


def test_goal_attribute():
    """Assert that the goal is deduced from the model's name."""
    # For classification tasks
    selector = FeatureSelector(strategy="SFM", solver="LGB_class")
    selector.fit(X_bin, y_bin)
    assert selector.goal == "classification"

    # For regression tasks
    selector = FeatureSelector(strategy="SFM", solver="LGB_reg")
    selector.fit(X_reg, y_reg)
    assert selector.goal == "regression"


def test_solver_parameter_invalid_value():
    """Assert that an error is raised when solver is unknown."""
    selector = FeatureSelector(strategy="RFE", solver="invalid")
    pytest.raises(ValueError, selector.fit, X_reg, y_reg)


def test_n_features_parameter():
    """Assert that an error is raised when n_features is invalid."""
    selector = FeatureSelector(strategy="SFM", solver="XGB_reg", n_features=0)
    pytest.raises(ValueError, selector.fit, X_reg, y_reg)


def test_max_frac_repeated_parameter():
    """Assert that an error is raised when max_frac_repeated is invalid."""
    selector = FeatureSelector(strategy=None, max_frac_repeated=1.1)
    pytest.raises(ValueError, selector.fit, X_reg, y_reg)


def test_max_correlation_parameter():
    """Assert that an error is raised when max_correlation is invalid."""
    selector = FeatureSelector(strategy=None, max_correlation=-0.2)
    pytest.raises(ValueError, selector.fit, X_reg, y_reg)


def test_error_y_is_None():
    """Assert that an error is raised when y is None for some strategies."""
    selector = FeatureSelector(strategy="univariate", solver=f_regression, n_features=9)
    pytest.raises(ValueError, selector.fit, X_reg)


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


def test_univariate_strategy_custom_solver():
    """Assert that the univariate strategy works for a custom solver."""
    selector = FeatureSelector("univariate", solver=f_regression, n_features=9)
    X = selector.fit_transform(X_reg, y_reg)
    assert X.shape[1] == 9
    assert set(selector.feature_importance) == set(X.columns)


def test_PCA_strategy():
    """Assert that the PCA strategy works as intended."""
    selector = FeatureSelector(strategy="PCA", n_features=0.7)
    X = selector.fit_transform(X_bin)
    assert X.shape[1] == 21


def test_PCA_components():
    """Assert that the PCA strategy creates components instead of features."""
    selector = FeatureSelector(strategy="PCA")
    X = selector.fit_transform(X_bin)
    assert "Component 1" in X.columns


def test_SFM_prefit_invalid_estimator():
    """Assert that an error is raised for an invalid estimator in SFM."""
    selector = FeatureSelector(
        strategy="SFM",
        solver=ExtraTreesClassifier(random_state=1).fit(X_class, y_class),
        n_features=8,
        random_state=1,
    )
    pytest.raises(ValueError, selector.fit, X_bin, y_bin)


def test_SFM_strategy_not_threshold():
    """Assert that if threshold is not specified, SFM selects n_features features."""
    selector = FeatureSelector(
        strategy="SFM",
        solver=ExtraTreesClassifier(random_state=1),
        n_features=16,
        random_state=1,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 16


def test_SFM_invalid_solver():
    """Assert that an error is raised when solver is invalid."""
    selector = FeatureSelector(strategy="SFM", solver="invalid", n_features=5)
    pytest.raises(ValueError, selector.fit_transform, X_bin, y_bin)


def test_SFM_strategy_fitted_solver():
    """Assert that the SFM strategy works when the solver is already fitted."""
    selector = FeatureSelector(
        strategy="SFM",
        solver=ExtraTreesClassifier(random_state=1).fit(X_bin, y_bin),
        n_features=7,
        random_state=1,
    )
    X = selector.fit_transform(X_bin)
    assert X.shape[1] == 7
    assert set(selector.feature_importance) == set(X.columns)


def test_SFM_strategy_not_fitted_solver():
    """Assert that the SFM strategy works when the solver is not fitted."""
    selector = FeatureSelector(
        strategy="SFM", solver=ExtraTreesClassifier(random_state=1), n_features=5
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 5
    assert set(selector.feature_importance) == set(X.columns)


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


def test_RFECV_strategy_before_pipeline_classification():
    """Assert that the RFECV strategy works before a fitted pipeline."""
    selector = FeatureSelector(
        strategy="RFECV",
        solver="RF_class",
        n_features=None,
        random_state=1,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 17
    assert set(selector.feature_importance) == set(X.columns)


def test_RFECV_strategy_before_pipeline_regression():
    """Assert that the RFECV strategy works before a fitted pipeline."""
    selector = FeatureSelector("RFECV", solver="RF_reg", n_features=16, random_state=1)
    X = selector.fit_transform(X_reg, y_reg)
    assert X.shape[1] == 10
    assert set(selector.feature_importance) == set(X.columns)


def test_SFS_strategy():
    """Assert that the SFS strategy works."""
    selector = FeatureSelector("SFS", solver="RF_reg", n_features=6, cv=3, random_state=1)
    X = selector.fit_transform(X_reg, y_reg)
    assert X.shape[1] == 6


def test_kwargs_parameter_threshold():
    """Assert that the kwargs parameter works as intended (add threshold)."""
    selector = FeatureSelector(
        strategy="SFM",
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
        strategy="PCA", solver="arpack", tol=0.001, n_features=12, random_state=1
    )
    X = selector.fit_transform(X_bin)
    assert X.shape[1] == 12


def test_kwargs_parameter_scoring():
    """Assert that the kwargs parameter works as intended (add scoring acronym)."""
    selector = FeatureSelector(
        strategy="RFECV",
        solver="rf_class",
        scoring="auc",
        n_features=12,
        random_state=1,
    )
    X = selector.fit_transform(X_bin, y_bin)
    assert X.shape[1] == 14
