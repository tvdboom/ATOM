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
import multiprocessing
from sklearn.metrics import get_scorer, f1_score

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.utils import merge, check_scaling
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg,
    X_bin_array, y_bin_array, X10_nan, X10_str, y10
    )


# << ================== Test api.py ================== >>

def test_log_is_none():
    """Assert that no logging file is created when log=None."""
    ATOMClassifier(X_bin, y_bin, log=None)
    assert not glob.glob('log.log')


def test_create_log_file():
    """Assert that a logging file is created when log is not None."""
    ATOMClassifier(X_bin, y_bin, log=FILE_DIR + 'log.log')
    assert glob.glob(FILE_DIR + 'log.log')


def test_log_file_ends_with_log():
    """Assert that the logging file always ends with log."""
    ATOMClassifier(X_bin, y_bin, log=FILE_DIR + 'logger')
    assert glob.glob(FILE_DIR + 'logger.log')


def test_log_file_named_auto():
    """Assert that when log='auto', an automatic logging file is created."""
    ATOMClassifier(X_bin, y_bin, log=FILE_DIR + 'auto')
    assert glob.glob(FILE_DIR + 'ATOM_logger_*')


def test_goal_assigning():
    """Assert that the goal attribute is assigned correctly."""
    atom = ATOMClassifier(X_bin, y_bin)
    assert atom.goal == 'classification'

    atom = ATOMRegressor(X_reg, y_reg)
    assert atom.goal == 'regression'


# << ================== Test __init__ ================== >>

def test_n_rows_parameter():
    """Assert that an error is raised when n_rows is <=0."""
    for n_rows in [0, -3]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, n_rows=n_rows)


def test_n_rows_parameter_too_large():
    """Assert that when n_rows is too large, whole X is selected."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=1e5)
    assert len(atom.dataset) == len(X_bin)


def test_test_size_parameter():
    """Assert that the test_size parameter is in correct range."""
    for size in [0., -3.1, 12.2]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, test_size=size)


def test_raise_one_target_value():
    """Assert that error raises when there is only 1 target value."""
    y = [1 for _ in range(len(y_bin))]  # All targets are equal to 1
    pytest.raises(ValueError, ATOMClassifier, X_bin, y)


def test_task_assigning():
    """Assert that the task attribute is assigned correctly."""
    atom = ATOMClassifier(X_bin, y_bin)
    assert atom.task == 'binary classification'

    atom = ATOMClassifier(X_class, y_class)
    assert atom.task == 'multiclass classification'

    atom = ATOMRegressor(X_reg, y_reg)
    assert atom.task == 'regression'


def test_merger_to_dataset():
    """Assert that the merger between X and y was successful."""
    atom = ATOMClassifier(X_bin, y_bin)
    merger = X_bin.merge(
        y_bin.astype(np.int64).to_frame(), left_index=True, right_index=True
        )

    # Order of rows can be different
    df1 = merger.sort_values(by=merger.columns.tolist())
    df1.reset_index(drop=True, inplace=True)
    df2 = atom.dataset.sort_values(by=atom.dataset.columns.tolist())
    df2.reset_index(drop=True, inplace=True)
    assert df1.equals(df2)


# << ================== Test properties ================== >>

def test_verbose_parameter():
    """Assert that the verbose parameter is in correct range."""
    for vb in [-2, 4]:
        pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, verbose=vb)


def test_n_jobs_maximum_cores():
    """Assert that value equals n_cores if maximum is exceeded."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=1000)
    assert atom.n_jobs == multiprocessing.cpu_count()


def test_n_jobs_is_zero():
    """Assert that when value=0, 1 core is used."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=0)
    assert atom.n_jobs == 1


def test_too_far_negative_n_jobs():
    """Assert that an error is raised when value too far negative."""
    pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, n_jobs=-1000)


def test_negative_n_jobs():
    """Assert that value is set correctly for negative values."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1)
    assert atom.n_jobs == multiprocessing.cpu_count()

    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-3)
    assert atom.n_jobs == multiprocessing.cpu_count() - 2


def test_random_state_parameter():
    """Assert the return of same results for two independent runs."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom.pipeline(['lr', 'lgb', 'pa'], 'f1', n_calls=8)
    atom2 = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom2.pipeline(['lr', 'lgb', 'pa'], 'f1', n_calls=8)

    assert atom.lr.score_test == atom2.lr.score_test
    assert atom.lgb.score_test == atom2.lgb.score_test
    assert atom.pa.score_test == atom2.pa.score_test


def test_random_state_setter():
    """Assert that an error is raised for a negative random_state."""
    pytest.raises(ValueError, ATOMClassifier, X_bin, y_bin, random_state=-1)


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
    atom = ATOMClassifier(X_bin, 'mean radius')
    assert atom.dataset.columns[-1] == 'mean radius'


# << ================== Test _split_dataset ================== >>

def test_n_rows_below_one():
    """Assert that a correct subset of the data is selected for n_rows<1."""
    n_rows = 0.5
    atom = ATOMClassifier(X_bin, y_bin, n_rows=n_rows)
    assert len(atom.dataset) == int(n_rows * len(X_bin))


def test_n_rows_above_one():
    """Assert that a correct subset of the data is selected for n_rows>1."""
    n_rows = 100
    atom = ATOMClassifier(X_bin, y_bin, n_rows=n_rows)
    assert len(atom.dataset) == n_rows


def test_dataset_is_shuffled():
    """Assert that the dataset is shuffled."""
    atom = ATOMClassifier(X_bin, y_bin)
    assert not atom.X.equals(X_bin)


def test_reset_index():
    """Assert that the indices are reset after the data is fractionated."""
    n_rows = 100
    atom = ATOMClassifier(X_bin, y_bin, n_rows=n_rows)
    assert list(atom.dataset.index) == list(range(n_rows))


def test_train_test_sizes():
    """Assert that the train and test set sizes are correctly set."""
    test_size = 0.13
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size)
    assert len(atom._train_idx) == int((1-test_size)*len(X_bin))
    assert len(atom._test_idx) == int(test_size*len(X_bin))+1


# << ================== Test report ================== >>

def test_creates_report():
    """Assert that the report attribute and file are created."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.report(n_rows=10, filename=FILE_DIR + 'report')
    assert hasattr(atom, 'report')
    assert glob.glob(FILE_DIR + 'report.html')


# << ================== Test results ================== >>

def test_is_fitted_results():
    """Assert that an error is raised when the ATOM class is not fitted."""
    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.results)


def test_error_unknown_metric():
    """Assert that an error is raised when an unknown metric is selected."""
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline(models='lgb', metric='r2')
    pytest.raises(ValueError, atom.results, 'unknown')


def test_error_invalid_metric():
    """Assert that an error is raised when an invalid metric is selected."""
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline(models='lgb', metric='r2')
    pytest.raises(ValueError, atom.results, 'average_precision')


def test_all_tasks():
    """Assert that the method works for all tasks."""
    # For binary classification
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(models=['lda', 'lgb'], metric='f1')
    atom.results()
    atom.results('jaccard')
    assert 1 == 1

    # For multiclass classification
    atom = ATOMClassifier(X_class, y_class)
    atom.pipeline(models=['pa', 'lgb'], metric='recall_macro')
    atom.results()
    atom.results('f1_micro')
    assert 2 == 2

    # For regression
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline(models='lgb', metric='neg_mean_absolute_error')
    atom.results()
    atom.results('neg_mean_poisson_deviance')
    assert 3 == 3


# << ================== Test clear ================== >>

def test_is_fitted_clear():
    """Assert that an error is raised when the ATOM class is not fitted."""
    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.clear)


def test_models_is_all():
    """Assert that the whole pipeline is cleared for models='all'."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['LR', 'LDA'])
    atom.clear('all')
    assert not atom.models
    assert not atom.winner
    assert not atom.scores


def test_models_is_str():
    """Assert that a single model is cleared."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['LR', 'LDA'])
    atom.clear('LDA')
    assert atom.models == ['LR']
    assert atom.winner is atom.LR
    assert len(atom.scores) == 1
    assert not hasattr(atom, 'LDA')


def test_models_is_sequence():
    """Assert that multiple models are cleared."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['LR', 'LDA', 'QDA'])
    atom.clear(['LDA', 'QDA'])
    assert atom.models == ['LR']
    assert atom.winner is atom.LR
    assert len(atom.scores) == 1


def test_clear_successive_halving():
    """Assert that clearing works for successive_halving pipelines."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.successive_halving(['LR', 'LDA', 'QDA'], bagging=3)
    atom.clear(['LR'])
    assert atom.scores[-1].empty
    assert atom.winner is atom.LDA


def test_clear_train_sizing():
    """Assert that clearing works for successive_halving pipelines."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.train_sizing(['LR', 'LDA', 'QDA'])
    atom.clear()
    assert not atom.models
    assert not atom.scores
    assert not atom.winner


# << ================== Test save ================== >>

def test_file_is_saved():
    """Assert that the pickle file is created."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.save(FILE_DIR + 'atom')
    assert glob.glob(FILE_DIR + 'atom.pkl')


# << ================== Test data cleaning methods ================== >>

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
    """Assert that the impute method imputes all missing values."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    assert all([atom.X[col].dtype.kind in 'ifu' for col in atom.X.columns])


def test_outliers():
    """Assert that the outliers method handles outliers in the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = len(atom.train)
    atom.outliers()
    assert len(atom.train) != length


def test_balance():
    """Assert that the balance method balances the training set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    length = (atom.y_train == 1).sum()
    atom.balance(undersample=0.8)
    assert (atom.y_train == 1).sum() != length


def test_feature_generation():
    """Assert that the feature_generation method creates extra features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_generation(n_features=2, generations=5, population=200)
    assert atom.dataset.shape[1] == X_bin.shape[1] + 2
    assert isinstance(atom.genetic_features, pd.DataFrame)


def test_feature_selection():
    """Assert that the feature_selection method removes features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='pca', n_features=8, max_correlation=0.8)
    assert atom.X.shape[1] == 8
    assert isinstance(atom.collinear, pd.DataFrame)


# << ================== Test _run_pipeline ================== >>

def test_invalid_models_parameter():
    """Assert that an error is raised for invalid or duplicate models."""
    atom = ATOMRegressor(X_reg, y_reg)
    pytest.raises(ValueError, atom.pipeline, models='test')
    pytest.raises(ValueError, atom.pipeline, models=['OLS', 'OLS'])


def test_invalid_task_models_parameter():
    """Assert that an error is raised for models with invalid tasks."""
    # Only classification
    atom = ATOMRegressor(X_reg, y_reg)
    pytest.raises(ValueError, atom.pipeline, models='LDA')

    # Only regression
    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(ValueError, atom.pipeline, models='OLS')


def test_skip_iter_parameter():
    """Assert that an error is raised for negative skip_iter."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.successive_halving, 'Tree', skip_iter=-1)


def test_n_calls_invalid_length():
    """Assert that an error is raised when len n_calls != models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.pipeline, 'Tree', n_calls=(3, 2))


def test_n_random_starts_invalid_length():
    """Assert that an error is raised when len n_random_starts != models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.pipeline, 'Tree', n_random_starts=(3, 2))


def test_n_calls_parameter_as_sequence():
    """Assert that n_calls as sequence works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['Tree', 'LGB'], n_calls=(3, 2), n_random_starts=1)
    assert len(atom.Tree.BO) == 3
    assert len(atom.LGB.BO) == 2


def test_n_random_starts_parameter_as_sequence():
    """Assert that n_random_starts as sequence works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['Tree', 'LGB'], n_calls=5, n_random_starts=(3, 1))
    assert (atom.Tree.BO['call'] == 'Random start').sum() == 3
    assert (atom.LGB.BO['call'] == 'Random start').sum() == 1


def test_kwargs_dimensions():
    """Assert that bo_kwargs['dimensions'] raises an error when wrong type."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    kwargs = {'dimensions': 3}
    pytest.raises(TypeError, atom.pipeline, ['Tree', 'LGB'], bo_kwargs=kwargs)


def test_default_metric_parameter():
    """Assert that the correct default metric is set per task."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline('LR')
    assert atom.metric.name == 'f1'

    atom = ATOMClassifier(X_class, y_class)
    atom.pipeline('LR')
    assert atom.metric.name == 'f1_weighted'

    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('OLS')
    assert atom.metric.name == 'r2'


def test_same_metric():
    """Assert that the default metric stays the same if already defined."""
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('OLS', metric='max_error')
    atom.pipeline('BR')
    assert atom.metric.name == 'max_error'


def test_invalid_metric_parameter():
    """Assert that an error is raised for an unknown metric."""
    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(ValueError, atom.pipeline, models='LDA', metric='unknown')


def test_function_metric_parameter():
    """Assert that a function metric works."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline('lr', metric=f1_score)
    assert not atom.errors


def test_scorer_metric_parameter():
    """Assert that a scorer metric works."""
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('ols', metric=get_scorer('neg_mean_squared_error'))
    assert not atom.errors


def test_invalid_train_sizes():
    """Assert than error is raised when element in train_sizes is >1."""
    atom = ATOMRegressor(X_reg, y_reg)
    pytest.raises(ValueError, atom.train_sizing, ['OLS'], train_sizes=[0.8, 2])


def test_scores_attribute():
    """Assert that the scores attribute has the right format."""
    atom = ATOMRegressor(X_reg, y_reg)

    # For a direct pipeline
    atom.pipeline('OLS')
    assert isinstance(atom.scores, pd.DataFrame)

    # For successive_halving
    atom.successive_halving(['OLS', 'BR', 'LGB', 'CatB'])
    assert isinstance(atom.scores, list)
    assert len(atom.scores) == 3

    # For successive_halving
    atom.train_sizing('OLS', train_sizes=[0.3, 0.6])
    assert isinstance(atom.scores, list)
    assert len(atom.scores) == 2


def test_exception_encountered():
    """Assert that exceptions are attached as attributes."""
    atom = ATOMClassifier(X_class, y_class)
    atom.pipeline(['BNB', 'LDA'], n_calls=3, n_random_starts=1)
    assert atom.BNB.error
    assert 'BNB' in atom.errors.keys()


def test_exception_removed_models():
    """Assert that models with exceptions are removed from self.models."""
    atom = ATOMClassifier(X_class, y_class)
    atom.pipeline(['BNB', 'LDA'], n_calls=3, n_random_starts=1)
    assert 'BNB' not in atom.models


def test_exception_not_subsequent_iterations():
    """Assert that models with exceptions are removed from following iters."""
    atom = ATOMClassifier(X_class, y_class)
    atom.train_sizing(['LR', 'LGB'], 'f1_macro', n_calls=3, n_random_starts=1)
    assert 'LGB' not in atom.scores[-1].index


def test_creation_subclasses_lowercase():
    """Assert that the model subclasses for lowercase are created."""
    atom = ATOMClassifier(X_class, y_class)
    atom.pipeline('LDA')
    assert hasattr(atom, 'lda')
    assert atom.LDA is atom.lda


def test_all_models_failed():
    """Assert than an error is raised when all models encountered errors."""
    atom = ATOMClassifier(X_class, y_class)
    pytest.raises(ValueError, atom.pipeline, 'BNB', n_calls=6)


def test_scores_columns():
    """Assert than self.scores has the right columns."""
    atom = ATOMClassifier(X_bin, y_bin)



# << ================== Test classmethods ================== >>

def test_set_style():
    """Assert that the set_style classmethod works as intended."""
    style = 'white'
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_style(style)
    assert ATOMClassifier.style == style


def test_set_palette():
    """Assert that the set_palette classmethod works as intended."""
    palette = 'Blues'
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_palette(palette)
    assert ATOMClassifier.palette == palette


def test_set_title_fontsize():
    """Assert that the set_title_fontsize classmethod works as intended."""
    title_fontsize = 21
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_title_fontsize(title_fontsize)
    assert ATOMClassifier.title_fontsize == title_fontsize


def test_set_label_fontsize():
    """Assert that the set_label_fontsize classmethod works as intended."""
    label_fontsize = 4
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_label_fontsize(label_fontsize)
    assert ATOMClassifier.label_fontsize == label_fontsize


def test_set_tick_fontsize():
    """Assert that the set_tick_fontsize classmethod works as intended."""
    tick_fontsize = 13
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_tick_fontsize(tick_fontsize)
    assert ATOMClassifier.tick_fontsize == tick_fontsize
