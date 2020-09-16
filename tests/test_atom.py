# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for atom.py

"""

# Import packages
import glob
import pytest
import numpy as np
import pandas as pd

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.utils import merge, check_scaling
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg,
    X_bin_array, y_bin_array, X10_nan, X10_str, y10, y10_str
    )


# Test __init__ ============================================================= >>

def test_n_rows_parameter():
    """Assert that an error is raised when n_rows is <=0."""
    pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, n_rows=0, random_state=1)


def test_n_rows_parameter_too_large():
    """Assert that when n_rows is too large, whole X is selected."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=1e5, n_jobs=2, random_state=1)
    assert len(atom.dataset) == len(X_bin)


def test_test_size_parameter():
    """Assert that the test_size parameter is in correct range."""
    for s in [0., -3.1, 12.2]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, test_size=s, random_state=1)


def test_input_is_prepared():
    """Assert that the _prepare_input method form BaseTransformer is run."""
    atom = ATOMClassifier(X_bin_array, y_bin_array, random_state=1)
    assert isinstance(atom.X, pd.DataFrame)
    assert isinstance(atom.y, pd.Series)


def test_raise_one_target_value():
    """Assert that error raises when there is only 1 target value."""
    y = [1 for _ in range(len(y_bin))]  # All targets are equal to 1
    pytest.raises(ValueError, ATOMClassifier, X_bin, y, random_state=1)


def test_task_assigning():
    """Assert that the task attribute is assigned correctly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.task == 'binary classification'

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.task == 'multiclass classification'

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.task == 'regression'


def test_mapping_assignment():
    """Assert that ATOM adopts mapping from the StandardCleaner class."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.mapping is atom.pipeline[0].mapping


def test_n_rows_fraction():
    """Assert that the n_rows selects a fraction of the dataset when <=1."""
    n_rows = 0.5
    atom = ATOMClassifier(X_bin, y_bin, n_rows=n_rows, random_state=1)
    assert len(atom.dataset) == int(n_rows * len(X_bin))


def test_n_rows_int():
    """Assert that the n_rows selects n rows of the dataset when >1."""
    n_rows = 400
    atom = ATOMClassifier(X_bin, y_bin, n_rows=n_rows, random_state=1)
    assert len(atom.dataset) == n_rows


def test_dataset_is_shuffled():
    """Assert that the dataset is shuffled before splitting."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert not X_bin.equals(atom.X)


def test_merger_to_dataset():
    """Assert that the merger between X and y was successful."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    merger = X_bin.merge(
        y_bin.astype(np.int64).to_frame(), left_index=True, right_index=True
        )

    # Order of rows can be different
    df1 = merger.sort_values(by=merger.columns.tolist())
    df1.reset_index(drop=True, inplace=True)
    df2 = atom.dataset.sort_values(by=atom.dataset.columns.tolist())
    df2.reset_index(drop=True, inplace=True)
    assert df1.equals(df2)


def test_reset_index():
    """Assert that the indices are reset for the whole dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert list(atom.dataset.index) == list(range(len(X_bin)))


def test_train_test_split():
    """Assert that the train/test split is made."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert len(atom.train) == round((1 - test_size) * len(X_bin)) + 1
    assert len(atom.test) == round(test_size * len(X_bin)) - 1


# Test utility properties ================================================== >>

def test_missing():
    """Assert that missing returns a series of missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    assert atom.missing.sum() == 1


def test_n_missing():
    """Assert that n_missing returns the number of missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    assert atom.n_missing == 1


def test_categorical():
    """Assert that categorical returns a list of categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.categorical == ['Feature 2']


def test_n_categorical():
    """Assert that n_categorical returns the number of categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    assert atom.categorical == ['Feature 2']


def test_scaled():
    """Assert that scaled returns if the dataset is scaled."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert not atom.scaled
    atom.scale()
    assert atom.scaled


# Test report =============================================================== >>

def test_creates_report():
    """Assert that the report attribute and file are created."""
    atom = ATOMClassifier(X_reg, y_reg, random_state=1)
    atom.report(n_rows=10, filename=FILE_DIR + 'report')
    assert hasattr(atom, 'report')
    assert glob.glob(FILE_DIR + 'report.html')


# Test transform ============================================================ >>

def test_transform_method():
    """ Assert that the transform method works as intended """
    atom = ATOMClassifier(X10_str, y10_str, random_state=1)
    atom.encode(max_onehot=None)
    atom.run('Tree')

    # With default arguments
    X_trans = atom.transform(X10_str)
    assert X_trans['Feature 2'].dtype.kind in 'ifu'

    # Changing arguments
    X_trans = atom.transform(X10_str, encode=False)
    assert X_trans['Feature 2'].dtype.kind not in 'ifu'


def test_verbose_raises_when_invalid():
    """Assert an error is raised for an invalid value of verbose."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.transform, X_bin, verbose=3)


def test_verbose_in_transform():
    """Assert that the verbosity of the transformed classes is changed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    _ = atom.transform(X_bin, verbose=2)
    assert atom.pipeline[0].verbose == 2


def test_pipeline_parameter():
    """Assert that the pipeline parameter is obeyed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers(max_sigma=1)
    X = atom.transform(X_bin, pipeline=[0])  # Only use StandardCleaner
    assert len(X) == len(X_bin)


def test_default_parameters():
    """Assert that outliers and balance are False by default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance()
    X = atom.transform(X_bin)
    assert len(X) == len(X_bin)


def test_parameters_are_obeyed():
    """Assert that only the transformations for the selected parameters are done."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers(max_sigma=1)
    X = atom.transform(X_bin, outliers=True)
    assert len(X) != len(X_bin)


def test_transform_with_y():
    """Assert that the transform method works when y is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers(max_sigma=2, include_target=True)
    X, y = atom.transform(X_bin, y_bin, outliers=True)
    assert len(y) < len(y_bin)


# Test data cleaning methods ================================================ >>

def test_ATOM_params_to_method():
    """Assert that the ATOM parameters are passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1)
    atom.scale()
    assert atom.pipeline[1].verbose == 1


def test_custom_params_to_method():
    """Assert that a custom parameter is passed to the method."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1)
    atom.scale(verbose=2)
    assert atom.pipeline[1].verbose == 2


def test_scale():
    """Assert that the scale method normalizes the features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    assert check_scaling(atom.dataset)


def test_impute():
    """Assert that the impute method imputes all missing values."""
    atom = ATOMClassifier(X10_nan, y10, random_state=1)
    atom.impute()
    assert atom.dataset.isna().sum().sum() == 0


def test_encode():
    """Assert that the encode method encodes all categorical columns."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    assert all([atom.X[col].dtype.kind in 'ifu' for col in atom.X.columns])


def test_outliers():
    """Assert that the outliers method handles outliers in the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = len(atom.train)
    atom.outliers()
    assert len(atom.train) != length


def test_balance_wrong_task():
    """Assert that an error is raised for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(PermissionError, atom.balance, oversample=0.7)


def test_balance():
    """Assert that the balance method balances the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = (atom.y_train == 0).sum()
    atom.balance()
    assert (atom.y_train == 0).sum() != length


def test_balance_mapping():
    """Assert that the balance method gets the mapping attribute from ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance()
    assert atom.pipeline[1].mapping == atom.mapping


def test_balance_attribute():
    """Assert that Balancer's estimator is attached to ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance(strategy='NearMiss')
    assert atom.nearmiss.__class__.__name__ == 'NearMiss'


# Test feature engineering methods ========================================== >>

def test_feature_generation():
    """Assert that the feature_generation method creates extra features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation(n_features=2, generations=5, population=200)
    assert atom.X.shape[1] == X_bin.shape[1] + 2


def test_feature_generation_attributes():
    """Assert that the attrs from feature_generation are passed to ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation('GFG', n_features=2, generations=5, population=200)
    assert hasattr(atom, 'symbolic_transformer')
    assert hasattr(atom, 'genetic_features')


def test_feature_selection_attrs():
    """Assert that feature_selection attaches only used attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='pca', n_features=8, max_correlation=0.8)
    assert hasattr(atom, 'collinear') and hasattr(atom, 'pca')


def test_default_solver_univariate():
    """Assert that the default solver is selected for strategy='univariate'."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='univariate', solver=None, n_features=8)
    assert atom.pipeline[1].solver.__name__ == 'f_classif'

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy='univariate', solver=None, n_features=8)
    assert atom.pipeline[1].solver.__name__ == 'f_regression'


def test_winner_solver_after_run():
    """Assert that the solver is the winning model after run."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run('LR')
    atom.feature_selection(
        strategy='SFM',
        solver=None,
        n_features=8,
        max_correlation=None
    )
    assert atom.pipeline[2].solver is atom.winner.estimator


def test_default_solver_from_task():
    """Assert that the solver is inferred from the task when a model is selected."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='rfe', solver='lgb', n_features=8)
    assert type(atom.pipeline[1].solver).__name__ == 'LGBMClassifier'

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy='rfe', solver='lgb', n_features=8)
    assert type(atom.pipeline[1].solver).__name__ == 'LGBMRegressor'


def test_default_scoring_RFECV():
    """Assert that the scoring for RFECV is ATOM's metric_ when exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr', metric='recall')
    atom.feature_selection(strategy='rfecv', solver='lgb', n_features=8)
    assert atom.pipeline[2].kwargs['scoring'].name == 'recall'


def test_plot_methods_attached():
    """Assert that the plot methods are attached to the ATOM instance."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='rfecv', solver='lgb', n_features=8)
    assert hasattr(atom, 'plot_rfecv')


# Test training methods ===================================================== >>

def test_errors_are_passed_to_ATOM():
    """Assert that the errors found in models are passed to ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'LGB'], n_calls=5, n_initial_points=(2, -1))
    assert atom.errors.get('LGB')


def test_mapping_is_passed_to_trainer():
    """Assert that the mapping attribute is passed to the trainer."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('LGB')
    assert atom.trainer.mapping is atom.mapping


def test_models_and_metric_are_updated():
    """Assert that the models and metric_ attributes are updated correctly."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(['LGB', 'CatB'], metric='max_error')
    assert atom.models == ['LGB', 'CatB']
    assert atom.metric == 'max_error'


def test_results_are_replaced():
    """Assert that the results are replaced for SuccessiveHalving and TrainSizing."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    atom.successive_halving('LGB')
    assert len(atom.results) == 1


def test_results_are_attached():
    """Assert that the results are attached for subsequent runs of Trainer."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    atom.run('LGB')
    assert len(atom.results) == 2


def test_errors_are_removed():
    """Assert that the errors are removed if subsequent runs are successful."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    # Invalid dimensions to create an error
    atom.run(['BNB', 'LGB'], bo_params={'dimensions': {'LGB': 2}})
    atom.run('LGB')  # Runs correctly
    assert not atom.errors  # Errors should be empty


def test_getattr_for_model_subclasses():
    """Assert that the model subclasses can be called through ATOM."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    assert atom.ols is atom.trainer.ols
    assert atom.OLS is atom.trainer.OLS


def test_model_subclasses_are_attached():
    """Assert that the model subclasses are attached to ATOM."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    assert hasattr(atom, 'OLS') and hasattr(atom, 'ols')


def test_pipeline_attr_is_attached():
    """Assert that the transform method is attached to the model subclasses."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    assert hasattr(atom.OLS, 'pipeline')
    assert hasattr(atom.ols, 'pipeline')


def test_errors_are_updated():
    """Assert that the found exceptions are updated in the errors attribute."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(['XGB', 'LGB'], n_calls=2, n_initial_points=2)  # No errors
    atom.run(['XGB', 'LGB'], n_calls=2, n_initial_points=(2, -1))  # Produces an error
    assert atom.errors.get('LGB')


def test_run_clear_results():
    """Assert that previous results are cleared if previous trainer wasn't run."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(['lr', 'lda'])
    atom.run('lr')
    assert 'LDA' not in atom.models


def test_assign_existing_metric():
    """Assert that the existing metric_ is assigned if rerun."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr', metric='recall')
    atom.run('lda')
    assert atom.metric == 'recall'


def test_raises_invalid_metric_consecutive_runs():
    """Assert that an error is raised for a different metric_."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr', metric='recall')
    pytest.raises(ValueError, atom.run, 'lda', metric='f1')


def test_call_Trainer():
    """Assert that the right class is called depending on the task."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr')
    assert type(atom.trainer).__name__ == 'TrainerClassifier'

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('ols')
    assert type(atom.trainer).__name__ == 'TrainerRegressor'


def test_call_SuccessiveHalving():
    """Assert that the right class is called depending on the task."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving('lr')
    assert type(atom.trainer).__name__ == 'SuccessiveHalvingClassifier'

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.successive_halving('ols')
    assert type(atom.trainer).__name__ == 'SuccessiveHalvingRegressor'


def test_call_TrainSizing():
    """Assert that the right class is called depending on the task."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing('lr')
    assert type(atom.trainer).__name__ == 'TrainSizingClassifier'

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.train_sizing('ols')
    assert type(atom.trainer).__name__ == 'TrainSizingRegressor'


# Test property getters ===================================================== >>

def test_dataset_setter():
    """Assert that the dataset setter changes the whole dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.dataset = merge(X_class, y_class)
    assert atom.dataset.shape == (len(X_class), X_class.shape[1]+1)


def test_train_setter():
    """Assert that the train setter changes the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train = atom.train.iloc[:100, :]
    assert atom.train.shape == (100, X_bin.shape[1]+1)


def test_test_setter():
    """Assert that the test setter changes the test set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.test = atom.test.iloc[:100, :]
    assert atom.test.shape == (100, X_bin.shape[1]+1)


def test_X_setter():
    """Assert that the X setter changes the feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X = atom.X.iloc[:, :10]
    assert atom.X.shape == (len(X_bin), 10)


def test_y_setter():
    """Assert that the y setter changes the target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y[0] == 1  # First value is 1 in original
    atom.y = [0] + list(y_bin.values[1:])
    assert atom.y[0] == 0  # First value changed to 0


def test_X_train_setter():
    """Assert that the X_train setter changes the training feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_train = atom.X_train
    new_X_train.iloc[0, 0] = 999
    atom.X_train = new_X_train
    assert atom.X_train.iloc[0, 0] == 999


def test_X_test_setter():
    """Assert that the X_test setter changes the test feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_test = atom.X_test
    new_X_test.iloc[0, 0] = 999
    atom.X_test = new_X_test
    assert atom.X_test.iloc[0, 0] == 999


def test_y_train_setter():
    """Assert that the y_train setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_train.iloc[0] == 1  # First value is 1 in original
    atom.y_train = [0] + list(atom.y_train.values[1:])
    assert atom.y_train.iloc[0] == 0  # First value changed to 0


def test_y_test_setter():
    """Assert that the y_test setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_test.iloc[0] == 0  # First value is 0 in original
    atom.y_test = [1] + list(atom.y_test.values[1:])
    assert atom.y_test.iloc[0] == 1  # First value changed to 1


def test_data_properties_to_df():
    """Assert that the data properties are converted to a df at setter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X = X_bin_array
    assert isinstance(atom.X, pd.DataFrame)


def test_data_properties_to_series():
    """Assert that the data properties are converted to a series at setter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.y = y_bin_array
    assert isinstance(atom.y, pd.Series)


def test_setter_error_unequal_rows():
    """Assert that an error is raised when the setter has unequal rows."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=r"number of rows"):
        atom.X_train = X_bin


def test_setter_error_unequal_index():
    """Assert that an error is raised when the setter has unequal indices."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=r"the same indices"):
        atom.y = pd.Series(y_bin_array, index=range(10, len(y_bin_array)+10))


def test_setter_error_unequal_columns():
    """Assert that an error is raised when the setter has unequal columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match='number of columns'):
        new_X = atom.train
        new_X.insert(0, 'new_column', 1)
        atom.train = new_X


def test_setter_error_unequal_column_names():
    """Assert that an error is raised with different column names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match='the same columns'):
        new_X = atom.train.drop(atom.train.columns[0], axis=1)
        new_X.insert(0, 'new_column', 1)
        atom.train = new_X


def test_setter_error_unequal_target_names():
    """Assert that an error is raised with different target names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match='the same name'):
        new_y_train = atom.y_train
        new_y_train.name = 'different_name'
        atom.y_train = new_y_train
