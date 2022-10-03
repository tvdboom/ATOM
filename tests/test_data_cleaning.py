# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM)

Author: Mavs
Description: Unit tests for data_cleaning.py

"""

import numpy as np
import pandas as pd
import pytest
from category_encoders.leave_one_out import LeaveOneOutEncoder
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from atom.data_cleaning import (
    Balancer, Cleaner, Discretizer, Encoder, Imputer, Normalizer, Pruner,
    Scaler,
)
from atom.utils import NotFittedError, check_scaling, to_df

from .conftest import (
    X10, X10_nan, X10_sn, X10_str, X10_str2, X_bin, X_class, X_idx, y10,
    y10_str, y_bin, y_class, y_idx,
)


# Test TransformerMixin ============================================ >>

def test_fit_transform():
    """Assert that the fit_transform method works as intended."""
    X_1 = Scaler().fit_transform(X_bin)
    X_2 = Scaler().fit(X_bin).transform(X_bin)
    pd.testing.assert_frame_equal(X_1, X_2)


def test_fit_transform_no_fit():
    """Assert that the fit_transform method works when no fit method."""
    X, y = Balancer().fit_transform(X_bin, y_bin)
    assert len(X) > len(X_bin)


# Test Balancer ==================================================== >>

def test_balancer_strategy_unknown_str():
    """Assert that an error is raised when strategy is unknown."""
    balancer = Balancer(strategy="invalid")
    with pytest.raises(ValueError, match=".*value for the strategy.*"):
        balancer.transform(X_bin, y_bin)


def test_balancer_strategy_invalid_estimator():
    """Assert that an error is raised when strategy is invalid."""
    balancer = Balancer(strategy=StandardScaler())
    with pytest.raises(TypeError, match=".*type for the strategy.*"):
        balancer.transform(X_bin, y_bin)


def test_balancer_custom_estimator():
    """Assert that the strategy can be a custom estimator."""
    balancer = Balancer(strategy=SMOTETomek)
    X, y = balancer.transform(X_bin, y_bin)
    assert len(X) != len(X_bin)

    balancer = Balancer(strategy=SMOTETomek())
    X, y = balancer.transform(X_bin, y_bin)
    assert len(X) != len(X_bin)


@pytest.mark.parametrize("strategy", ["allknn", "tomeklinks", "svmsmote", "smoteenn"])
def test_balancers(strategy):
    """Assert that balancer estimators work as intended."""
    balancer = Balancer(strategy=strategy, sampling_strategy="all")
    X, y = balancer.transform(X_bin, y_bin)
    assert len(X) != len(X_bin)


def test_balancer_kwargs():
    """Assert that kwargs can be passed to the estimator."""
    balancer = Balancer(strategy="SMOTE", k_neighbors=12)
    balancer.transform(X_class, y_class)
    assert balancer.smote.get_params()["k_neighbors"] == 12


def test_oversampling_numerical_index():
    """Assert that new samples have an increasing int index."""
    X, y = Balancer(strategy="smote").transform(X_bin, y_bin)
    assert list(X.index) == list(range(len(X)))


def test_oversampling_string_index():
    """Assert that new samples have a new index."""
    X, y = Balancer(strategy="smote").transform(X_idx, y_idx)
    assert X.index[-1] == f"smote_{len(X) - len(X_idx)}"


def test_undersampling_keeps_indices():
    """Assert that indices are kept after transformation."""
    X, y = Balancer(strategy="nearmiss").transform(X_bin, y_bin)
    assert list(X.index) != list(range(len(X)))


def test_combinations_numerical_index():
    """Assert that new samples have an increasing int index."""
    X, y = Balancer(strategy="smoteenn").transform(X_bin, y_bin)
    assert not all(idx in X.index for idx in X_bin.index)  # Samples were dropped
    assert max(X.index) > max(X_bin.index)  # Samples were added


def test_combinations_string_index():
    """Assert that new samples have a new index."""
    X, y = Balancer(strategy="smotetomek").transform(X_idx, y_idx)
    assert not all(idx in X.index for idx in X_bin.index)  # Samples were dropped
    assert len(X.index.str.startswith("smotetomek") > 0)  # Samples were added


def test_balancer_attach_attribute():
    """Assert that the estimator is attached as attribute to the class."""
    balancer = Balancer(strategy="smote")
    balancer.transform(X_bin, y_bin)
    assert hasattr(balancer, "smote")


# Test Cleaner ==================================================== >>

def test_cleaner_drop_invalid_column_type():
    """Assert that invalid columns types are dropped for string input."""
    X = X_bin.copy()
    X["datetime_col"] = pd.to_datetime(X["mean radius"])  # Datetime column
    X = Cleaner(drop_types="datetime64[ns]").fit_transform(X)
    assert "datetime_col" not in X.columns


def test_cleaner_drop_invalid_column_list_types():
    """Assert that invalid columns types are dropped for list input."""
    X = X_bin.copy()
    X["datetime_col"] = pd.to_datetime(X["mean radius"])  # Datetime column
    X["string_col"] = [str(i) for i in range(len(X))]  # String column
    cleaner = Cleaner(drop_types=["datetime64[ns]", "object"])
    X = cleaner.fit_transform(X)
    assert "datetime_col" not in X.columns
    assert "string_col" not in X.columns


def test_cleaner_strip_categorical_features():
    """Assert that categorical features are stripped from blank spaces."""
    X = X_bin.copy()
    X["string_col"] = [" " + str(i) + " " for i in range(len(X))]
    X = Cleaner().fit_transform(X)
    series = pd.Series([str(i) for i in range(len(X))], name="string_col")
    pd.testing.assert_series_equal(X["string_col"], series)


def test_cleaner_strip_ignores_nan():
    """Assert that the stripping ignores missing values."""
    X = Cleaner().fit_transform(X10_sn)
    assert X.isna().sum().sum() == 1


def test_cleaner_drop_duplicate_rows():
    """Assert that that duplicate rows are removed."""
    X = Cleaner(drop_duplicates=True).fit_transform(X10)
    assert len(X) == 7


def test_cleaner_drop_missing_target():
    """Assert that rows with missing values in the target column are dropped."""
    y = y_bin.copy()
    length = len(X_bin)
    y[0], y[21], y[41] = np.NaN, 99, np.inf  # Set missing to target column
    cleaner = Cleaner()
    cleaner.missing.append(99)
    _, y = cleaner.fit_transform(X_bin, y)
    assert length == len(y) + 3


def test_cleaner_label_encoder_target_column():
    """Assert that the label-encoder for the target column works."""
    X, y = Cleaner().fit_transform(X10, y10_str)
    assert np.all((y == 0) | (y == 1))


def test_cleaner_inverse_transform():
    """Assert that the inverse_transform method works."""
    cleaner = Cleaner().fit(y=y10_str)
    y = cleaner.inverse_transform(y=cleaner.transform(y=y10_str))
    pd.testing.assert_series_equal(pd.Series(y10_str, name="target"), y)


def test_cleaner_target_mapping_binary():
    """Assert that the mapping attribute is set for binary tasks."""
    cleaner = Cleaner().fit(y=y10_str)
    assert cleaner.mapping == {"n": 0, "y": 1}


def test_cleaner_target_mapping_multiclass():
    """Assert that the mapping attribute is set for multiclass tasks."""
    cleaner = Cleaner().fit(y=y_class)
    assert cleaner.mapping == {"0": 0, "1": 1, "2": 2}


# Test Discretizer ================================================= >>

def test_strategy_parameter_discretizer():
    """Assert that the strategy parameter is set correctly."""
    discretizer = Discretizer(strategy="invalid")
    with pytest.raises(ValueError, match=".*value for the strategy parameter.*"):
        discretizer.fit(X_bin)


def test_invalid_bins_missing_column():
    """Assert that an error is raised when a column is missing."""
    discretizer = Discretizer(strategy="uniform", bins={"invalid": 5})
    with pytest.raises(ValueError, match=".*not found in the dictionary.*"):
        discretizer.fit(X_bin)


def test_invalid_bins_custom_strategy():
    """Assert that an error is raised when bins are not a sequence."""
    discretizer = Discretizer(strategy="custom", bins=5)
    with pytest.raises(TypeError, match=".*a sequence of bin edges.*"):
        discretizer.fit(X_bin)


def test_invalid_length_labels():
    """Assert that an error is raised when len(bins) != len(labels)."""
    discretizer = Discretizer(strategy="custom", bins=[5, 10, 15], labels=["label"])
    with pytest.raises(ValueError, match=".*length of the bins does not match.*"):
        discretizer.fit(X_bin)


def test_invalid_bins_to_column_length():
    """Assert that an error is raised when len(bins) != len(columns)."""
    discretizer = Discretizer(strategy="uniform", bins=[5, 10])
    with pytest.raises(ValueError, match=".*length of the bins does not match.*"):
        discretizer.fit(X_bin)


@pytest.mark.parametrize("strategy", ["uniform", "quantile", "kmeans"])
def test_discretizer_strategies(strategy):
    """Assert that custom binning can be performed."""
    discretizer = Discretizer(strategy=strategy, bins=5)
    X = discretizer.fit_transform(X_bin)
    assert all(X[col].dtype.name == "object" for col in X)


def test_custom_strategy():
    """Assert that custom binning can be performed."""
    discretizer = Discretizer(strategy="custom", bins=[0, 25])
    X = discretizer.fit_transform(X_bin)
    assert X["mean texture"].unique().tolist() == ["(0, 25]", "(25, inf]"]


def test_bins_is_sequence():
    """Assert that bins can be provided as sequence."""
    discretizer = Discretizer(strategy="uniform", bins=[5, 6, 7])
    X = discretizer.fit_transform(X_bin.iloc[:, :3])
    assert X[X.columns[0]].nunique() == 5
    assert X[X.columns[1]].nunique() == 6
    assert X[X.columns[2]].nunique() == 7


def test_bins_is_dict():
    """Assert that bins can be provided as dict."""
    discretizer = Discretizer(
        strategy="uniform",
        bins={X_bin.columns[0]: 5, X_bin.columns[1]: 6, X_bin.columns[2]: 7},
    )
    X = discretizer.fit_transform(X_bin.iloc[:, :3])
    assert X[X.columns[0]].nunique() == 5
    assert X[X.columns[1]].nunique() == 6
    assert X[X.columns[2]].nunique() == 7


def test_labels_non_custom_strategy():
    """Assert that custom labels can be added to strategy != custom."""
    discretizer = Discretizer(strategy="uniform", bins=3, labels=["l1", "l2", "l3"])
    X = discretizer.fit_transform(X_bin)
    assert X["mean texture"].unique().tolist() == ["l1", "l2", "l3"]


def test_labels_custom_strategy():
    """Assert that custom labels can be added to the custom strategy."""
    discretizer = Discretizer(
        strategy="custom",
        bins=[10, 20],
        labels={"mean texture": ["l1", "l2", "l3"]},
    )
    X = discretizer.fit_transform(X_bin)
    assert X["mean texture"].unique().tolist() == ["l2", "l3", "l1"]


# Test Encoder ===================================================== >>

def test_strategy_parameter_encoder():
    """Assert that the strategy parameter is set correctly."""
    encoder = Encoder(strategy="invalid")
    with pytest.raises(ValueError, match=".*value for the strategy.*"):
        encoder.fit(X10_str, y10)


def test_strategy_with_encoder_at_end():
    """Assert that the strategy works with Encoder at the end of the string."""
    encoder = Encoder(strategy="TargetEncoder", max_onehot=None)
    encoder.fit(X10_str, y10)
    assert encoder._encoders["x2"].__class__.__name__ == "TargetEncoder"


def test_max_onehot_parameter():
    """Assert that the max_onehot parameter is set correctly."""
    encoder = Encoder(max_onehot=-2)
    with pytest.raises(ValueError, match=".*value for the max_onehot.*"):
        encoder.fit(X10_str, y10)


def test_rare_to_value_parameter():
    """Assert that the rare_to_value parameter is set correctly."""
    encoder = Encoder(rare_to_value=-2)
    with pytest.raises(ValueError, match=".*value for the rare_to_value.*"):
        encoder.fit(X10_str, y10)


@pytest.mark.parametrize("rare_to_value", [3, 0.3])
def test_rare_to_value(rare_to_value):
    """Assert that the other values are created when encoding."""
    encoder = Encoder(max_onehot=5, rare_to_value=rare_to_value)
    X = encoder.fit_transform(X10_str, y10)
    assert "x2_rare" in X.columns


def test_encoder_strategy_invalid_estimator():
    """Assert that an error is raised when strategy is invalid."""
    encoder = Encoder(strategy=RandomForestClassifier())
    with pytest.raises(TypeError, match=".*type for the strategy.*"):
        encoder.fit_transform(X10_str, y10)


def test_encoder_custom_estimator():
    """Assert that the strategy can be a custom estimator."""
    encoder = Encoder(strategy=LeaveOneOutEncoder, max_onehot=None)
    X = encoder.fit_transform(X10_str, y10)
    assert X.at[0, "x2"] != "a"

    encoder = Encoder(strategy=LeaveOneOutEncoder(), max_onehot=None)
    X = encoder.fit_transform(X10_str, y10)
    assert X.at[0, "x2"] != "a"


def test_encoder_check_is_fitted():
    """Assert that an error is raised if the instance is not fitted."""
    pytest.raises(NotFittedError, Encoder().transform, X_bin, y_bin)


def test_missing_values_are_propagated():
    """Assert that missing values are propagated."""
    encoder = Encoder(max_onehot=None)
    assert np.isnan(encoder.fit_transform(X10_sn, y10).iat[0, 2])


def test_unknown_classes_are_imputed():
    """Assert that unknown classes are imputed."""
    encoder = Encoder()
    encoder.fit(["a", "b", "b", "a"])
    assert encoder.transform(["c"]).iat[0, 0] == -1.0


def test_ordinal_encoder():
    """Assert that the Ordinal-encoder works as intended."""
    encoder = Encoder(max_onehot=None)
    X = encoder.fit_transform(X10_str2, y10)
    assert np.all((X["x2"] == 0) | (X["x2"] == 1))
    assert list(encoder.mapping) == ["x2", "x3"]


def test_ordinal_features():
    """Assert that ordinal features are encoded."""
    encoder = Encoder(max_onehot=None, ordinal={"x2": ["b", "a", "c"]})
    X = encoder.fit_transform(X10_str2, y10)
    assert X.iat[0, 2] == 1 and X.iat[2, 2] == 0


def test_one_hot_encoder():
    """Assert that the OneHot-encoder works as intended."""
    encoder = Encoder(max_onehot=4)
    X = encoder.fit_transform(X10_str, y10)
    assert "x2_c" in X.columns


@pytest.mark.parametrize("strategy", ["HelmertEncoder", "SumEncoder"])
def test_all_encoder_types(strategy):
    """Assert that encoding estimators work as intended."""
    encoder = Encoder(strategy=strategy, max_onehot=None)
    X = encoder.fit_transform(X10_str, y10)
    assert all(X[col].dtype.kind in "ifu" for col in X)


def test_kwargs_parameters():
    """Assert that the kwargs parameter works as intended."""
    encoder = Encoder(strategy="LeaveOneOut", max_onehot=None, sigma=0.5)
    encoder.fit(X10_str, y10)
    assert encoder._encoders["x2"].get_params()["sigma"] == 0.5


# Test Imputer ===================================================== >>

def test_strat_num_parameter():
    """Assert that the strat_num parameter is set correctly."""
    imputer = Imputer(strat_num="invalid")
    with pytest.raises(ValueError, match=".*Unknown strategy for the strat_num.*"):
        imputer.fit(X_bin, y_bin)


def test_invalid_max_nan_rows():
    """Assert that an error is raised for invalid max_nan_rows."""
    imputer = Imputer(max_nan_rows=-2)
    with pytest.raises(ValueError, match=".*value for the max_nan_rows.*"):
        imputer.fit(X_bin, y_bin)


def test_invalid_max_nan_cols():
    """Assert that an error is raised for invalid max_nan_cols."""
    imputer = Imputer(max_nan_cols=-5)
    with pytest.raises(ValueError, match=".*value for the max_nan_cols.*"):
        imputer.fit(X_bin, y_bin)


def test_imputer_check_is_fitted():
    """Assert that an error is raised if the instance is not fitted."""
    pytest.raises(NotFittedError, Imputer().transform, X_bin, y_bin)


@pytest.mark.parametrize("missing", [None, np.NaN, np.inf, -np.inf, 99])
def test_imputing_all_missing_values_numeric(missing):
    """Assert that all missing values are imputed in numeric columns."""
    X = [[missing, 1, 1], [2, 5, 2], [4, missing, 1], [2, 1, 1]]
    y = [1, 1, 0, 0]
    imputer = Imputer(strat_num="mean")
    imputer.missing.append(99)
    X, y = imputer.fit_transform(X, y)
    assert X.isna().sum().sum() == 0


@pytest.mark.parametrize("missing", ["", "?", "NaN", "NA", "nan", "inf"])
def test_imputing_all_missing_values_categorical(missing):
    """Assert that all missing values are imputed in categorical columns."""
    X = [[missing, "a", "a"], ["b", "c", missing], ["b", "a", "c"], ["c", "a", "a"]]
    y = [1, 1, 0, 0]
    imputer = Imputer(strat_cat="most_frequent")
    X, y = imputer.fit_transform(X, y)
    assert X.isna().sum().sum() == 0


@pytest.mark.parametrize("max_nan_rows", [5, 0.5])
def test_rows_too_many_nans(max_nan_rows):
    """Assert that rows with too many missing values are dropped."""
    X = X_bin.copy()
    for i in range(5):  # Add 5 rows with all NaN values
        X.loc[len(X)] = [np.nan for _ in range(X.shape[1])]
    y = [np.random.randint(2) for _ in range(len(X))]
    imputer = Imputer(
        strat_num="mean",
        strat_cat="most_frequent",
        max_nan_rows=max_nan_rows,
    )
    X, y = imputer.fit_transform(X, y)
    assert len(X) == 569  # Original size
    assert X.isna().sum().sum() == 0


def test_all_rows_too_many_nans():
    """Assert that an error is raised when all rows have too many nans."""
    X = X_bin.copy()
    X["only nans"] = np.NaN
    imputer = Imputer(strat_num="mean", strat_cat="most_frequent", max_nan_rows=0.01)
    with pytest.raises(ValueError, match=".*All rows contain.*"):
        imputer.fit_transform(X, y_bin)


@pytest.mark.parametrize("max_nan_cols", [20, 0.5])
def test_cols_too_many_nans(max_nan_cols):
    """Assert that columns with too many missing values are dropped."""
    X = X_bin.copy()
    for i in range(5):  # Add 5 cols with all NaN values
        X["col " + str(i)] = [np.nan for _ in range(X.shape[0])]
    imputer = Imputer(
        strat_num="mean",
        strat_cat="most_frequent",
        max_nan_cols=max_nan_cols,
    )
    X, y = imputer.fit_transform(X, y_bin)
    assert len(X.columns) == 30  # Original number of columns
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_drop():
    """Assert that imputing drop for numerical values works."""
    imputer = Imputer(strat_num="drop")
    X, y = imputer.fit_transform(X10_nan, y10)
    assert len(X) == 8
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_number():
    """Assert that imputing a number for numerical values works."""
    imputer = Imputer(strat_num=3.2)
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iat[0, 0] == 3.2
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_knn():
    """Assert that imputing numerical values with KNNImputer works."""
    imputer = Imputer(strat_num="knn")
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iat[0, 0] == pytest.approx(2.577778, rel=1e-6, abs=1e-12)
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_mean():
    """Assert that imputing the mean for numerical values works."""
    imputer = Imputer(strat_num="mean")
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iat[0, 0] == pytest.approx(2.577778, rel=1e-6, abs=1e-12)
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_median():
    """Assert that imputing the median for numerical values works."""
    imputer = Imputer(strat_num="median")
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iat[0, 0] == 3
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_most_frequent():
    """Assert that imputing the most_frequent for numerical values works."""
    imputer = Imputer(strat_num="most_frequent")
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iat[0, 0] == 3
    assert X.isna().sum().sum() == 0


def test_imputing_non_numeric_string():
    """Assert that imputing a string for non-numerical values works."""
    imputer = Imputer(strat_cat="missing")
    X, y = imputer.fit_transform(X10_sn, y10)
    assert X.iat[0, 2] == "missing"
    assert X.isna().sum().sum() == 0


def test_imputing_non_numeric_drop():
    """Assert that the drop strategy for non-numerical works."""
    imputer = Imputer(strat_cat="drop")
    X, y = imputer.fit_transform(X10_sn, y10)
    assert len(X) == 9
    assert X.isna().sum().sum() == 0


def test_imputing_non_numeric_most_frequent():
    """Assert that the most_frequent strategy for non-numerical works."""
    imputer = Imputer(strat_cat="most_frequent")
    X, y = imputer.fit_transform(X10_sn, y10)
    assert X.iat[0, 2] == "d"
    assert X.isna().sum().sum() == 0


# Test Normalizer ======================================================= >>

def test_normalizer_check_is_fitted():
    """Assert that an error is raised when not fitted."""
    pytest.raises(NotFittedError, Normalizer().transform, X_bin)


def test_normalizer_invalid_strategy():
    """Assert that an error is raised when strategy is invalid."""
    normalizer = Normalizer(strategy="invalid")
    with pytest.raises(ValueError, match=".*value for the strategy.*"):
        normalizer.fit(X_bin)


@pytest.mark.parametrize("strategy", ["yeojohnson", "boxcox", "quantile"])
def test_normalizer_all_strategies(strategy):
    """Assert that all strategies work as intended."""
    normalizer = Normalizer(strategy=strategy)
    normalizer.fit_transform(X10)


def test_normalizer_categorical_columns():
    """Assert that categorical columns are ignored."""
    X = Normalizer().fit_transform(X10_str)
    assert X["x2"].dtype.kind == "O"


def test_normalizer_inverse_transform():
    """Assert that the inverse_transform method works."""
    normalizer = Normalizer().fit(X_bin)
    X = normalizer.inverse_transform(X_bin)
    assert X.shape == X_bin.shape


def test_normalizer_inverse_categorical_columns():
    """Assert that categorical columns are ignored."""
    X = Normalizer().fit(X10_str).inverse_transform(X10_str)
    assert X["x2"].dtype.kind == "O"


def test_normalizer_y_is_ignored():
    """Assert that the y parameter is ignored if provided."""
    X_1 = Normalizer().fit_transform(X_bin, y_bin)
    X_2 = Normalizer().fit_transform(X_bin)
    pd.testing.assert_frame_equal(X_1, X_2)


def test_normalizer_kwargs():
    """Assert that kwargs can be passed to the estimator."""
    X = Normalizer(strategy="yeojohnson", standardize=False).fit_transform(X_bin)
    assert not check_scaling(X)


def test_normalizer_ignores_categorical_columns():
    """Assert that categorical columns are ignored."""
    X = X_bin.copy()
    X.insert(1, "categorical_col_1", ["a" for _ in range(len(X))])
    X = Normalizer().fit_transform(X)
    assert list(X[X.columns.values[1]]) == ["a" for _ in range(len(X))]


def test_normalizer_attach_attribute():
    """Assert that the estimator is attached as attribute to the class."""
    normalizer = Normalizer(strategy="quantile")
    normalizer.fit_transform(X_bin)
    assert hasattr(normalizer, "quantile")


# Test Pruner ====================================================== >>

def test_invalid_strategy_parameter():
    """Assert that an error is raised for an invalid strategy parameter."""
    pruner = Pruner(strategy="invalid")
    with pytest.raises(ValueError, match=".*value for the strategy.*"):
        pruner.transform(X_bin)


def test_invalid_method_for_non_z_score():
    """Assert that an error is raised for an invalid method and strat combination."""
    pruner = Pruner(strategy="iforest", method="min_max")
    with pytest.raises(ValueError, match=".*accepts another method.*"):
        pruner.transform(X_bin)


def test_invalid_method_parameter():
    """Assert that an error is raised for an invalid method parameter."""
    pruner = Pruner(method="invalid")
    with pytest.raises(ValueError, match=".*value for the method parameter.*"):
        pruner.transform(X_bin)


def test_invalid_max_sigma_parameter():
    """Assert that an error is raised for an invalid max_sigma parameter."""
    pruner = Pruner(max_sigma=0)
    with pytest.raises(ValueError, match=".*value for the max_sigma.*"):
        pruner.transform(X_bin)


def test_max_sigma_functionality():
    """Assert that the max_sigma parameter works as intended."""
    # Test 3 different values for sigma and number of rows they drop
    X_1 = Pruner(max_sigma=1).fit_transform(X_bin)
    X_2 = Pruner(max_sigma=4).fit_transform(X_bin)
    X_3 = Pruner(max_sigma=8).fit_transform(X_bin)
    assert len(X_1) < len(X_2) < len(X_3)


def test_kwargs_parameter_pruner():
    """Assert that the kwargs are passed to the strategy estimator."""
    pruner = Pruner(strategy="iForest", n_estimators=50)
    pruner.transform(X10)
    assert pruner.iforest.get_params()["n_estimators"] == 50


def test_drop_pruner():
    """Assert that rows with outliers are dropped when strategy="drop"."""
    X = Pruner(method="drop", max_sigma=2).transform(X10)
    assert len(X) + 2 == len(X10)


def test_min_max_pruner():
    """Assert that the method works as intended when strategy="min_max"."""
    X = Pruner(method="min_max", max_sigma=2).transform(X10)
    assert X.iat[3, 0] == 0.23  # Max of column
    assert X.iat[5, 1] == 2  # Min of column


def test_value_pruner():
    """Assert that the method works as intended when strategy=value."""
    X = Pruner(method=-99, max_sigma=2).transform(X10)
    assert X.iat[3, 0] == -99
    assert X.iat[5, 1] == -99


def test_categorical_cols_are_ignored():
    """Assert that categorical columns are returned untouched."""
    X, y = Pruner(method="min_max", max_sigma=2).transform(X10_str, y10)
    pd.testing.assert_series_equal(X["x1"], to_df(X10_str)["x1"])


def test_drop_outlier_in_target():
    """Assert that method works as intended for target columns as well."""
    X, y = Pruner(max_sigma=2, include_target=True).transform(X10, y10)
    assert len(y) + 2 == len(y10)


@pytest.mark.parametrize("strategy", ["iforest", "ee", "lof", "svm", "dbscan"])
def test_pruner_strategies(strategy):
    """Assert that all estimator requiring strategies work."""
    pruner = Pruner(strategy=strategy)
    X, y = pruner.transform(X_bin, y_bin)
    assert len(X) < len(X_bin)
    assert hasattr(pruner, strategy.lower())


def test_multiple_strategies():
    """Assert that selecting multiple strategies work."""
    pruner = Pruner(strategy=["zscore", "lof", "ee", "iforest"])
    X, y = pruner.transform(X_bin, y_bin)
    assert len(X) < len(X_bin)
    assert all(hasattr(pruner, attr) for attr in ("lof", "ee", "iforest"))


def test_kwargs_one_strategy():
    """Assert that kwargs can be provided for one strategy."""
    pruner = Pruner(strategy="iforest", n_estimators=100)
    pruner.transform(X_bin, y_bin)
    assert pruner.iforest.get_params()["n_estimators"] == 100


def test_kwargs_multiple_strategies():
    """Assert that kwargs can be provided for multiple strategies."""
    pruner = Pruner(["svm", "lof"], svm={"kernel": "poly"}, lof={"n_neighbors": 10})
    pruner.transform(X_bin, y_bin)
    assert pruner.svm.get_params()["kernel"] == "poly"
    assert pruner.lof.get_params()["n_neighbors"] == 10


def test_pruner_attach_attribute():
    """Assert that the estimator is attached as attribute to the class."""
    pruner = Pruner(strategy="iforest")
    pruner.transform(X_bin)
    assert hasattr(pruner, "iforest")


# Test Scaler ====================================================== >>

def test_scaler_check_is_fitted():
    """Assert that an error is raised when not fitted."""
    pytest.raises(NotFittedError, Scaler().transform, X_bin)


def test_scaler_invalid_strategy():
    """Assert that an error is raised when strategy is invalid."""
    scaler = Scaler(strategy="invalid")
    with pytest.raises(ValueError, match=".*value for the strategy.*"):
        scaler.fit(X_bin)


@pytest.mark.parametrize("strategy", ["standard", "minmax", "maxabs", "robust"])
def test_scaler_all_strategies(strategy):
    """Assert that all strategies work as intended."""
    scaler = Scaler(strategy=strategy)
    scaler.fit_transform(X_bin)


def test_scaler_categorical_columns():
    """Assert that categorical columns are ignored."""
    X = Scaler().fit_transform(X10_str)
    assert X["x2"].dtype.kind == "O"


def test_scaler_y_is_ignored():
    """Assert that the y parameter is ignored if provided."""
    X_1 = Scaler().fit_transform(X_bin, y_bin)
    X_2 = Scaler().fit_transform(X_bin)
    pd.testing.assert_frame_equal(X_1, X_2)


def test_scaler_kwargs():
    """Assert that kwargs can be passed to the estimator."""
    X = Scaler(strategy="minmax", feature_range=(1, 2)).fit_transform(X_bin)
    assert min(X.iloc[:, 0]) >= 1 and max(X.iloc[:, 0]) <= 2


def test_scaler_return_scaled_dataset():
    """Assert that the returned dataframe is indeed scaled."""
    X = Scaler().fit_transform(X_bin)
    assert check_scaling(X)


def test_scaler_inverse_transform():
    """Assert that the inverse_transform method works."""
    scaler = Scaler().fit(X_bin)
    X_transformed = scaler.transform(X_bin)
    X_original = scaler.inverse_transform(X_transformed)
    pd.testing.assert_frame_equal(X_bin, X_original)


def test_scaler_inverse_categorical_columns():
    """Assert that categorical columns are ignored."""
    X = Scaler().fit(X10_str).inverse_transform(X10_str)
    assert X["x2"].dtype.kind == "O"


def test_scaler_ignores_categorical_columns():
    """Assert that categorical columns are ignored."""
    X = X_bin.copy()
    X.insert(1, "categorical_col_1", ["a" for _ in range(len(X))])
    X = Scaler().fit_transform(X)
    assert list(X[X.columns.values[1]]) == ["a" for _ in range(len(X))]


def test_scaler_attach_attribute():
    """Assert that the estimator is attached as attribute to the class."""
    scaler = Scaler(strategy="robust")
    scaler.fit_transform(X_bin)
    assert hasattr(scaler, "robust")
