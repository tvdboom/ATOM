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
    X_bin_array, y_bin_array, X10_nan, X10_str, y10
    )


# Test __init__ ============================================================= >>

def test_n_rows_parameter():
    """Assert that an error is raised when n_rows is <=0."""
    pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, n_rows=0, random_state=1)


def test_n_rows_parameter_too_large():
    """Assert that when n_rows is too large, whole X is selected."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=1e5, random_state=1)
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
    assert atom.mapping is atom.standard_cleaner.mapping


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
    assert len(atom.train) == int((1 - test_size) * len(X_bin))
    assert len(atom.test) == round(test_size * len(X_bin))


# Test properties =========================================================== >>

def test_results_getter():
    """Assert that the results property doesn't return columns with NaNs."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom.run('lr', 'f1', n_calls=0)
    assert 'mean_bagging' not in atom.results


def test_dataset_property():
    """Assert that the dataset property returns the whole dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dataset.shape == (len(X_bin), X_bin.shape[1]+1)


def test_dataset_setter():
    """Assert that the dataset setter changes the whole dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.dataset = merge(X_class, y_class)
    assert atom.dataset.shape == (len(X_class), X_class.shape[1]+1)


def test_train_property():
    """Assert that the train property returns the training set."""
    test_size = 0.3
    train_size = 1 - test_size
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.train.shape == (int(train_size*len(X_bin)), X_bin.shape[1]+1)


def test_train_setter():
    """Assert that the train setter changes the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train = atom.train.iloc[:100, :]
    assert atom.train.shape == (100, X_bin.shape[1]+1)


def test_test_property():
    """Assert that the test property returns the test set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.test.shape == (int(test_size*len(X_bin))+1, X_bin.shape[1]+1)


def test_test_setter():
    """Assert that the test setter changes the test set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.test = atom.test.iloc[:100, :]
    assert atom.test.shape == (100, X_bin.shape[1]+1)


def test_X_property():
    """Assert that the X property returns the feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.X.shape == (len(X_bin), X_bin.shape[1])


def test_X_setter():
    """Assert that the X setter changes the feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.X = atom.X.iloc[:, :10]
    assert atom.X.shape == (len(X_bin), 10)


def test_y_property():
    """Assert that the y property returns the target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y.shape == (len(y_bin),)


def test_y_setter():
    """Assert that the y setter changes the target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y[0] == 1  # First value is 1 in original
    atom.y = [0] + list(y_bin.values[1:])
    assert atom.y[0] == 0  # First value changed to 0


def test_X_train_property():
    """Assert that the X_train property returns the training feature set."""
    test_size = 0.3
    train_size = 1 - test_size
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.X_train.shape == (int(train_size*len(X_bin)), X_bin.shape[1])


def test_X_train_setter():
    """Assert that the X_train setter changes the training feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_train = atom.X_train
    new_X_train.iloc[0, 0] = 999
    atom.X_train = new_X_train
    assert atom.X_train.iloc[0, 0] == 999


def test_X_test_property():
    """Assert that the X_test property returns the test feature set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.X_test.shape == (int(test_size*len(X_bin))+1, X_bin.shape[1])


def test_X_test_setter():
    """Assert that the X_test setter changes the test feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    new_X_test = atom.X_test
    new_X_test.iloc[0, 0] = 999
    atom.X_test = new_X_test
    assert atom.X_test.iloc[0, 0] == 999


def test_y_train_property():
    """Assert that the y_train property returns the training target column."""
    test_size = 0.3
    train_size = 1 - test_size
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.y_train.shape == (int(train_size*len(X_bin)),)


def test_y_train_setter():
    """Assert that the y_train setter changes the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y_train.iloc[0] == 1  # First value is 1 in original
    atom.y_train = [0] + list(atom.y_train.values[1:])
    assert atom.y_train.iloc[0] == 0  # First value changed to 0


def test_y_test_property():
    """Assert that the y_test property returns the training target column."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.y_test.shape == (int(test_size*len(X_bin))+1,)


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


def test_target_is_last_column():
    """Assert that target is placed as the last column of the dataset."""
    atom = ATOMClassifier(X_bin, 'mean radius', random_state=1)
    assert atom.dataset.columns[-1] == 'mean radius'


# Test report =============================================================== >>

def test_creates_report():
    """Assert that the report attribute and file are created."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.report(n_rows=10, filename=FILE_DIR + 'report')
    assert hasattr(atom, 'report')
    assert glob.glob(FILE_DIR + 'report.html')


# Test transform ============================================================ >>

def test_verbose_raises_when_invalid():
    """Assert an error is raised for an invalid value of verbose."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.transform, X_bin, verbose=3)


def test_verbose_in_transform():
    """Assert that the verbosity of the transformed classes is changed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    _ = atom.transform(X_bin, verbose=2)
    assert atom.standard_cleaner.verbose == 2


def test_parameters_are_obeyed():
    """Assert that only the transformations for the selected parameters are done."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers(max_sigma=1)
    X = atom.transform(X_bin, outliers=False)
    assert len(X) == len(X_bin)


def test_transform_with_y():
    """Assert that the transform method works when y is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.outliers(max_sigma=2, include_target=True)
    X, y = atom.transform(X_bin, y_bin, outliers=True)
    assert len(y) < len(y_bin)


# Test clear ================================================================ >>

def test_models_is_all():
    """Assert that the whole pipeline is cleared for models='all'."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'LDA'])
    atom.clear('all')
    assert not (atom.models or atom.metric or atom.trainer or atom.winner)
    assert atom.results.empty


def test_models_is_str():
    """Assert that a single model is cleared."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'LDA'])
    atom.clear('LDA')
    assert atom.models == ['LR']
    assert atom.winner is atom.LR
    assert len(atom.results) == 1
    assert not hasattr(atom, 'LDA')


def test_models_is_sequence():
    """Assert that multiple models are cleared."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'LDA', 'QDA'])
    atom.clear(['LDA', 'QDA'])
    assert atom.models == ['LR']
    assert atom.winner is atom.LR
    assert len(atom.results) == 1


def test_clear_successive_halving():
    """Assert that clearing works for successive_halving pipelines."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(['LR', 'LDA', 'QDA'], bagging=3)
    atom.clear(['LR'])
    assert 'LR' not in atom.results.index.get_level_values(1)
    assert atom.winner is atom.LDA


def test_clear_train_sizing():
    """Assert that clearing works for successive_halving pipelines."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing(['LR', 'LDA', 'QDA'])
    atom.clear()
    assert not (atom.models or atom.metric or atom.trainer or atom.winner)
    assert atom.results.empty


# Test save ================================================================= >>

def test_file_is_saved():
    """Assert that the pickle file is created."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.save(FILE_DIR + 'atom')
    assert glob.glob(FILE_DIR + 'atom.pkl')


# Test data cleaning methods ================================================ >>

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
    pytest.raises(RuntimeError, atom.balance, oversample=0.7)


def test_balance():
    """Assert that the balance method balances the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = (atom.y_train == 1).sum()
    atom.balance(undersample=0.8)
    assert (atom.y_train == 1).sum() != length


def test_balance_mapping():
    """Assert that the balance method gets the mapping attribute from ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance(undersample=0.8)
    assert atom.balancer.mapping == atom.mapping


def test_feature_generation():
    """Assert that the feature_generation method creates extra features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation(n_features=2, generations=5, population=200)
    assert atom.dataset.shape[1] == X_bin.shape[1] + 2


def test_feature_generation_attributes():
    """Assert that the attrs from feature_generation are passed to ATOM."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation(n_features=2, generations=5, population=200)
    assert hasattr(atom, 'genetic_algorithm')
    assert hasattr(atom, 'genetic_features')


def test_feature_selection_attrs():
    """Assert that the feature_selection attaches only used attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='pca', n_features=8, max_correlation=0.8)
    assert hasattr(atom, 'collinear')
    assert hasattr(atom, 'pca')
    assert not hasattr(atom, 'rfe')


def test_default_solver_univariate():
    """Assert that the default solver is selected for strategy='univariate'."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='univariate', solver=None, n_features=8)
    assert atom.feature_selector.solver.__name__ == 'f_classif'

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy='univariate', solver=None, n_features=8)
    assert atom.feature_selector.solver.__name__ == 'f_regression'


def test_winner_solver_after_run():
    """Assert that the solver is the winning model after run."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr')
    atom.feature_selection(strategy='sfm', solver=None, n_features=8)
    assert atom.feature_selector.solver is atom.winner.best_model_fit


def test_default_solver_from_task():
    """Assert that the solver is inferred from the task when a model is selected."""
    # For classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='rfe', solver='lgb', n_features=8)
    assert type(atom.feature_selector.solver).__name__ == 'LGBMClassifier'

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.feature_selection(strategy='rfe', solver='lgb', n_features=8)
    assert type(atom.feature_selector.solver).__name__ == 'LGBMRegressor'


def test_default_scoring_RFECV():
    """Assert that the scoring for RFECV is ATOM's metric when exists."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr', metric='recall')
    atom.feature_selection(strategy='rfecv', solver='lgb', n_features=8)
    assert atom.feature_selector.kwargs['scoring'].name == 'recall'


def test_plot_methods_attached():
    """Assert that the plot methods are attached to the ATOM instance."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='rfecv', solver='lgb', n_features=8)
    assert hasattr(atom, 'plot_rfecv')


# Test training methods ===================================================== >>

def test_errors_are_passed_to_ATOM():
    """Assert that the errors found in models are passed to ATOM."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(ValueError, atom.run, 'LGB', metric='f1_weighted')
    assert 'LGB' in atom.errors


def test_methods_are_passed_to_ATOM():
    """Assert that the plot and transformation methods are passed to ATOM."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('LGB')
    assert hasattr(atom, 'plot_gains')
    assert hasattr(atom, 'predict')
    assert hasattr(atom, 'outcome')


def test_mapping_is_passed_to_trainer():
    """Assert that the mapping attribute is passed to the trainer."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('LGB')
    assert atom.trainer.mapping is atom.mapping


def test_models_and_metric_are_updated():
    """Assert that the models and metric attributes are updated correctly."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(['LGB', 'CatB'], metric='max_error')
    assert atom.models == ['LGB', 'CatB']
    assert atom.metric.name == 'max_error'


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
    atom.run(['LDA', 'LGB'], n_calls=2, n_random_starts=2)  # LGB creates an error
    atom.run('LGB', bo_kwargs={'cv': 1})  # Runs correctly
    assert not atom.errors  # Errors should be empty


def test_model_subclasses_are_attached():
    """Assert that the model subclasses are attached to ATOM."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    assert hasattr(atom, 'OLS')
    assert hasattr(atom, 'ols')


def test_transform_method_is_attached():
    """Assert that the transform method is attached to the model subclasses."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    assert hasattr(atom.OLS, 'transform')
    assert hasattr(atom.ols, 'transform')


def test_new_winner():
    """Assert that the winner attribute changes."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('LGB')
    winner = atom.winner.name
    atom.run('OLS')
    assert atom.winner.name != winner


def test_errors_are_updated():
    """Assert that the found exceptions are updated in the errors attribute."""
    atom = ATOMClassifier(X_reg, y_reg, random_state=1)
    atom.run('LGB', bo_kwargs={'cv': 1})
    atom.run(['XGB', 'LGB'], n_calls=2, n_random_starts=2)
    assert atom.errors.get('LGB')


def test_run_clear_results():
    """Assert that previous results are cleared if previous trainer wasn't run."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(['lr', 'lda'])
    atom.run('lr')
    assert 'lda' not in atom.models


def test_assign_default_metric():
    """Assert that the default metric is assigned when metric=None."""
    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr', metric=None)
    assert atom.metric.name == 'f1'

    # For multiclass classification tasks
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run('lr', metric=None)
    assert atom.metric.name == 'f1_weighted'

    # For regression tasks
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('ols', metric=None)
    assert atom.metric.name == 'r2'


def test_assign_existing_metric():
    """Assert that the existing metric is assigned if rerun."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr', metric='recall')
    atom.run('lda')
    assert atom.metric.name == 'recall'


def test_raises_invalid_metric_consecutive_runs():
    """Assert that an error is raised for a different metric."""
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
