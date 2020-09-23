# coding: utf-8

"""Automated Tool for Optimized Modelling (ATOM)

Author: tvdboom
Description: Unit tests for data_cleaning.py

"""

# Import Packages =========================================================== >>

# Standard packages
import pytest
import numpy as np
import pandas as pd

# Own modules
from atom.utils import ENCODER_TYPES, BALANCER_TYPES, NotFittedError, check_scaling
from atom.data_cleaning import (
    Scaler, StandardCleaner, Imputer, Encoder, Outliers, Balancer
    )
from .utils import (
    X_bin, y_bin, X_class, y_class, X10, X10_nan, X10_str, X10_str2,
    X10_sn, y10, y10_str
    )


# Test BaseCleaner ========================================================== >>

def test_fit_transform():
    """Assert that the fit_transform method works as intended."""
    X_1 = Scaler().fit_transform(X_bin)
    X_2 = Scaler().fit(X_bin).transform(X_bin)
    assert X_1.equals(X_2)


def test_fit_transform_no_fit():
    """Assert that the fit_transform method works when no fit method."""
    X = X_bin.copy()
    X['test_column'] = 1  # Create a column with minimum cardinality
    X_1 = StandardCleaner().fit_transform(X)
    X_2 = StandardCleaner().transform(X)
    assert X_1.equals(X_2)


# Test Scaler =============================================================== >>

def test_check_is_fitted():
    """Assert that an error is raised when not fitted."""
    pytest.raises(NotFittedError, Scaler().transform, X_bin)


def test_y_is_ignored():
    """Assert that the y parameter is ignored if provided."""
    X = Scaler().fit_transform(X_bin, y_bin)
    X_2 = Scaler().fit_transform(X_bin)
    assert X.equals(X_2)


def test_return_df():
    """Assert that a pd.DataFrame is returned, not a np.array."""
    X = Scaler().fit_transform(X_bin)
    assert isinstance(X, pd.DataFrame)


def test_return_correct_columns():
    """Assert that the returned dataframe has the original columns."""
    X = Scaler().fit_transform(X_bin)
    assert list(X.columns) == list(X_bin.columns)


def test_return_correct_index():
    """Assert that the returned dataframe has the original indices."""
    X = Scaler().fit_transform(X_bin)
    assert list(X.index) == list(X_bin.index)


def test_return_scaled_dataset():
    """Assert that the returned dataframe is indeed scaled."""
    X = Scaler().fit_transform(X_bin)
    assert check_scaling(X)


# Test StandardCleaner ====================================================== >>

def test_drop_invalid_column_type():
    """Assert that invalid columns types are dropped for string input."""
    X = X_bin.copy()
    X['datetime_col'] = pd.to_datetime(X['mean radius'])  # Datetime column
    X = StandardCleaner(prohibited_types='datetime64[ns]').transform(X)
    assert 'datetime_col' not in X.columns


def test_drop_invalid_column_list_types():
    """Assert that invalid columns types are dropped for list input."""
    X = X_bin.copy()
    X['datetime_col'] = pd.to_datetime(X['mean radius'])  # Datetime column
    X['string_col'] = [str(i) for i in range(len(X))]  # String column
    X = StandardCleaner(prohibited_types=['datetime64[ns]', 'object'],
                        maximum_cardinality=False).transform(X)
    assert 'datetime_col' not in X.columns
    assert 'string_col' not in X.columns


def test_strip_categorical_features():
    """Assert that categorical features are stripped from blank spaces."""
    X = X_bin.copy()
    X['string_col'] = [' ' + str(i) + ' ' for i in range(len(X))]
    X = StandardCleaner(maximum_cardinality=False).transform(X)
    assert X['string_col'].equals(pd.Series([str(i) for i in range(len(X))]))


def test_strip_ignores_nan():
    """Assert that the stripping ignores missing values."""
    X = StandardCleaner(maximum_cardinality=False).transform(X10_sn)
    assert X.isna().sum().sum() == 1


def test_drop_maximum_cardinality():
    """Assert that categorical columns with maximum cardinality are dropped."""
    X = X_bin.copy()
    # Create column with all different values
    X['invalid_column'] = [str(i) for i in range(len(X))]
    X = StandardCleaner().transform(X)
    assert 'invalid_column' not in X.columns


def test_drop_minimum_cardinality():
    """Assert that columns with minimum cardinality are dropped."""
    X = X_bin.copy()
    X['invalid_column'] = 2.3  # Create column with only one value
    X = StandardCleaner().transform(X)
    assert 'invalid_column' not in X.columns


def test_drop_rows_nan_target():
    """Assert that self.dataset drops rows with NaN in target column."""
    y = y_bin.copy()
    length = len(X_bin)  # Save number of rows
    y[0], y[21] = np.NaN, np.NaN  # Set NaN to target column for 2 rows
    _, y = StandardCleaner().transform(X_bin, y)
    assert length == len(y) + 2


def test_label_encoder_target_column():
    """Assert that the label-encoder for the target column works."""
    X, y = StandardCleaner().transform(X10, y10_str)
    assert np.all((y == 0) | (y == 1))


def test_target_mapping():
    """Assert that the mapping attribute is set correctly."""
    cleaner = StandardCleaner()

    # For binary classification
    cleaner.transform(X10, y10_str)
    assert cleaner.mapping == dict(n=0, y=1)

    # For multiclass classification
    cleaner.transform(X_class, y_class)
    assert cleaner.mapping == {'0': 0, '1': 1, '2': 2}


# Test Imputer ============================================================== >>

def test_strat_num_parameter():
    """Assert that the strat_num parameter is set correctly."""
    imputer = Imputer(strat_num='invalid')
    pytest.raises(ValueError, imputer.fit, X_bin, y_bin)


def test_min_frac_rows_parameter():
    """Assert that the min_frac_rows parameter is set correctly."""
    imputer = Imputer(min_frac_rows=1.0)
    pytest.raises(ValueError, imputer.fit, X_bin, y_bin)


def test_min_frac_cols_parameter():
    """Assert that the min_frac_cols parameter is set correctly."""
    imputer = Imputer(min_frac_cols=5.2)
    pytest.raises(ValueError, imputer.fit, X_bin, y_bin)


def test_missing_parameter_is_string():
    """Assert that the missing parameter works when it's a string."""
    X = [[4, 1, 2], [3, 1, 2], ['r', 'a', 'b'], [2, 1, 1]]
    y = [1, 0, 0, 1]
    imputer = Imputer(strat_num='drop', strat_cat='drop', missing='a')
    X, y = imputer.fit_transform(X, y)
    assert X.isna().sum().sum() == 0


def test_missing_parameter_adds_extra_values():
    """Assert that the missing parameter adds obligatory values."""
    X = [[1, 1, 2], [None, 1, 2], ['a', 'a', 'b'], [2, np.inf, 1]]
    y = [1, 1, 0, 0]
    impute = Imputer(strat_num='drop',
                     strat_cat='drop',
                     missing=['O', 'N'],
                     min_frac_rows=0.1,
                     min_frac_cols=0.1)
    X, y = impute.fit_transform(X, y)
    assert X.isna().sum().sum() == 0


def test_is_fitted():
    """Assert that an error is raised if class is not fitted."""
    pytest.raises(NotFittedError, Imputer().transform, X_bin, y_bin)


def test_imputing_all_missing_values_numeric():
    """Assert that all missing values are imputed in numeric columns."""
    for v in [None, np.NaN, np.inf, -np.inf]:
        X = [[v, 1, 1], [2, 5, 2], [4, v, 1], [2, 1, 1]]
        y = [1, 1, 0, 0]
        imputer = Imputer(strat_num='mean')
        X, y = imputer.fit_transform(X, y)
        assert X.isna().sum().sum() == 0


def test_imputing_all_missing_values_categorical():
    """Assert that all missing values are imputed in categorical columns."""
    for v in ['', '?', 'NA', 'nan', 'inf']:
        X = [[v, '1', '1'], ['2', '5', v], ['2', '1', '3'], ['3', '1', '1']]
        y = [1, 1, 0, 0]
        imputer = Imputer(strat_cat='most_frequent')
        X, y = imputer.fit_transform(X, y)
        assert X.isna().sum().sum() == 0


def test_rows_too_many_nans():
    """Assert that rows with too many NaN values are dropped."""
    X = X_bin.copy()
    for i in range(5):  # Add 5 rows with all NaN values
        X.loc[len(X)] = [np.nan for _ in range(X.shape[1])]
    y = [np.random.randint(2) for _ in range(len(X))]
    impute = Imputer(strat_num='mean', strat_cat='most_frequent')
    X, y = impute.fit_transform(X, y)
    assert len(X) == 569  # Original size
    assert X.isna().sum().sum() == 0


def test_cols_too_many_nans():
    """Assert that columns with too many NaN values are dropped."""
    X = X_bin.copy()
    for i in range(5):  # Add 5 cols with all NaN values
        X['col ' + str(i)] = [np.nan for _ in range(X.shape[0])]
    impute = Imputer(strat_num='mean', strat_cat='most_frequent')
    X, y = impute.fit_transform(X, y_bin)
    assert len(X.columns) == 30  # Original number of columns
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_drop():
    """Assert that imputing drop for numerical values works."""
    imputer = Imputer(strat_num='drop')
    X, y = imputer.fit_transform(X10_nan, y10)
    assert len(X) == 9
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_number():
    """Assert that imputing a number for numerical values works."""
    imputer = Imputer(strat_num=3.2, min_frac_cols=0.1, min_frac_rows=0.1)
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iloc[0, 0] == 3.2
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_knn():
    """Assert that imputing numerical values with KNNImputer works."""
    imputer = Imputer(strat_num='knn')
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iloc[0, 0] == pytest.approx(2.577778, rel=1e-6, abs=1e-12)
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_mean():
    """Assert that imputing the mean for numerical values works."""
    imputer = Imputer(strat_num='mean')
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iloc[0, 0] == pytest.approx(2.577778, rel=1e-6, abs=1e-12)
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_median():
    """Assert that imputing the median for numerical values works."""
    imputer = Imputer(strat_num='median')
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iloc[0, 0] == 3
    assert X.isna().sum().sum() == 0


def test_imputing_numeric_most_frequent():
    """Assert that imputing the most_frequent for numerical values works."""
    imputer = Imputer(strat_num='most_frequent')
    X, y = imputer.fit_transform(X10_nan, y10)
    assert X.iloc[0, 0] == 3
    assert X.isna().sum().sum() == 0


def test_imputing_non_numeric_string():
    """Assert that imputing a string for non-numerical values works."""
    imputer = Imputer(strat_cat='missing')
    X, y = imputer.fit_transform(X10_sn, y10)
    assert X.iloc[0, 2] == 'missing'
    assert X.isna().sum().sum() == 0


def test_imputing_non_numeric_drop():
    """Assert that the drop strategy for non-numerical works."""
    imputer = Imputer(strat_cat='drop')
    X, y = imputer.fit_transform(X10_sn, y10)
    assert len(X) == 9
    assert X.isna().sum().sum() == 0


def test_imputing_non_numeric_most_frequent():
    """Assert that the most_frequent strategy for non-numerical works."""
    imputer = Imputer(strat_cat='most_frequent')
    X, y = imputer.fit_transform(X10_sn, y10)
    assert X.iloc[0, 2] == 'd'
    assert X.isna().sum().sum() == 0


# Test Encoder ============================================================== >>

def test_strategy_parameter_encoder():
    """Assert that the strategy parameter is set correctly."""
    encoder = Encoder(strategy='invalid')
    pytest.raises(ValueError, encoder.fit, X_bin, y_bin)


def test_strategy_with_encoder_at_end():
    """Assert that the strategy works with Encoder at the end of the string."""
    encoder = Encoder(strategy='TargetEncoder', max_onehot=None)
    encoder.fit(X10_str, y10)
    assert encoder._encoders['Feature 2'].__class__.__name__ == 'TargetEncoder'


def test_max_onehot_parameter():
    """Assert that the max_onehot parameter is set correctly."""
    encoder = Encoder(max_onehot=-2)
    pytest.raises(ValueError, encoder.fit, X_bin, y_bin)


def test_frac_to_other_parameter():
    """Assert that the frac_to_other parameter is set correctly."""
    encoder = Encoder(frac_to_other=2.2)
    pytest.raises(ValueError, encoder.fit, X_bin, y_bin)


def test_frac_to_other():
    """Assert that the other values are created when encoding."""
    encoder = Encoder(max_onehot=5, frac_to_other=0.3)
    X = encoder.fit_transform(X10_str, y10)
    assert 'Feature 2_other' in X.columns


def test_raise_missing_fit():
    """Assert that an error is raised when there are missing values during fit."""
    encoder = Encoder(max_onehot=None)
    pytest.raises(ValueError, encoder.fit_transform, X10_sn, y10)


def test_raise_missing_transform():
    """Assert that an error is raised when there are missing values during trans."""
    encoder = Encoder(max_onehot=None)
    encoder.fit(X10_str, y10)
    pytest.raises(ValueError, encoder.transform, X10_sn, y10)


def test_label_encoder():
    """Assert that the Label-encoder works as intended."""
    encoder = Encoder(max_onehot=None)
    X = encoder.fit_transform(X10_str2, y10)
    assert np.all((X['Feature 2'] == 0) | (X['Feature 2'] == 1))


def test_one_hot_encoder():
    """Assert that the OneHot-encoder works as intended."""
    encoder = Encoder(max_onehot=4)
    X = encoder.fit_transform(X10_str, y10)
    assert 'Feature 2_c' in X.columns


@pytest.mark.parametrize('strategy', ENCODER_TYPES)
def test_all_encoder_types(strategy):
    """Assert that all estimators work as intended."""
    encoder = Encoder(strategy=strategy, max_onehot=None)
    X = encoder.fit_transform(X10_str, y10)
    assert all([X[col].dtype.kind in 'ifu' for col in X])


def test_kwargs_parameters():
    """Assert that the kwargs parameter works as intended."""
    encoder = Encoder(strategy='LeaveOneOut', max_onehot=None, sigma=0.5)
    encoder.fit(X10_str, y10)
    assert encoder._encoders['Feature 2'].get_params()['sigma'] == 0.5


# Test Outliers ============================================================= >>

def test_invalid_strategy_parameter():
    """Assert that the strategy parameter is set correctly."""
    outliers = Outliers(strategy='invalid')
    pytest.raises(ValueError, outliers.transform, X_bin)


def test_max_sigma_parameter():
    """Assert that the max_sigma parameter is set correctly."""
    outliers = Outliers(strategy='min_max', max_sigma=0)
    pytest.raises(ValueError, outliers.transform, X_bin)


def test_max_sigma_functionality():
    """Assert that the max_sigma parameter works as intended."""
    # Test 3 different values for sigma and number of rows they drop
    X_1 = Outliers(max_sigma=1).fit_transform(X_bin)
    X_2 = Outliers(max_sigma=4).fit_transform(X_bin)
    X_3 = Outliers(max_sigma=8).fit_transform(X_bin)
    assert len(X_1) < len(X_2) < len(X_3)


def test_drop_outliers():
    """Assert that rows with outliers are dropped when strategy='drop'."""
    X = Outliers(strategy='drop', max_sigma=2).transform(X10)
    assert len(X) + 2 == len(X10)


def test_min_max_outliers():
    """Assert that the method works as intended when strategy='min_max'."""
    X = Outliers(strategy='min_max', max_sigma=2).transform(X10)
    assert X.iloc[3, 0] == 0.23  # Max of column
    assert X.iloc[5, 1] == 2  # Min of column


def test_value_outliers():
    """Assert that the method works as intended when strategy=value."""
    X = Outliers(strategy=-99, max_sigma=2).transform(X10)
    assert X.iloc[3, 0] == -99
    assert X.iloc[5, 1] == -99


def test_categorical_cols_are_ignores():
    """Assert that categorical columns are returned untouched."""
    Feature_2 = np.array(X10_str)[:, 2]
    X, y = Outliers(strategy='min_max', max_sigma=2).transform(X10_str, y10)
    assert [i == j for i, j in zip(X['Feature 2'], Feature_2)]


def test_drop_outlier_in_target():
    """Assert that method works as intended for target columns as well."""
    X, y = Outliers(max_sigma=2, include_target=True).transform(X10, y10)
    assert len(y) + 2 == len(y10)


# Test Balancer ============================================================= >>

def test_strategy_parameter_balancer():
    """Assert that an error is raised when strategy is invalid."""
    balancer = Balancer(strategy='invalid')
    pytest.raises(ValueError, balancer.transform, X_bin, y_bin)


def test_kwargs_parameter():
    """Assert that the kwargs are passed to the estimator."""
    balancer = Balancer(strategy='SMOTE', k_neighbors=12)
    balancer.transform(X_class, y_class)
    assert balancer.smote.get_params()['k_neighbors'] == 12


@pytest.mark.parametrize('strategy', [i for i in BALANCER_TYPES if i != 'smotenc'])
def test_all_balancers(strategy):
    """Assert that all estimators work as intended."""
    balancer = Balancer(strategy=strategy, sampling_strategy='all')
    X, y = balancer.transform(X_bin, y_bin)
    assert len(X) != len(X_bin)


@pytest.mark.parametrize('sampling', [1.0, 0.7, 'minority', 'not majority', 'all'])
def test_sampling_binary(sampling):
    """Assert that the oversampling method works for binary tasks."""
    X, y = Balancer(sampling_strategy=sampling).transform(X_bin, y_bin)
    assert (y == 1).sum() != (y_class == 1).sum()


@pytest.mark.parametrize('sampling', ['minority', 'not majority', 'all'])
def test_sampling_multiclass(sampling):
    """Assert that the oversampling method works for multiclass tasks."""
    X, y = Balancer(sampling_strategy=sampling).transform(X_class, y_class)
    assert (y == 2).sum() != (y_class == 2).sum()
