# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for basemodel.py

"""

# Import packages
import glob
import pytest
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from skopt.callbacks import TimerCallback
from skopt.space.space import Integer

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.utils import check_scaling
from .utils import FILE_DIR, X_bin, y_bin, X_reg, y_reg, X10_str, y10


# Test bayesian_optimization ================================================ >>

def test_n_random_starts_lower_1():
    """Assert than an error is raised when n_random_starts<1."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'LDA'], n_calls=5, n_random_starts=(2, -1))
    assert atom.errors.get('LDA')


def test_n_calls_lower_n_random_starts():
    """Assert than an error is raised when n_calls<n_random_starts."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'LDA'], n_calls=(5, 2), n_random_starts=(2, 3))
    assert atom.errors.get('LDA')


def test_callbacks_class():
    """Assert that custom callbacks works as intended for a class."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR', n_calls=5, bo_params={'callbacks': TimerCallback()})
    assert not atom.errors


def test_callbacks_sequence():
    """Assert that custom callbacks works as intended for a sequence."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR', n_calls=5, bo_params={'callbacks': [TimerCallback()]})
    assert not atom.errors


def test_invalid_max_time():
    """Assert than an error is raised when max_time<0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(
        RuntimeError, atom.run, 'LR', n_calls=5, bo_params={'max_time': -1})


def test_invalid_delta_x():
    """Assert than an error is raised when delta_x<0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, 'LR', n_calls=5, bo_params={'delta_x': -1})


def test_invalid_delta_y():
    """Assert than an error is raised when delta_y<0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, 'LR', n_calls=5, bo_params={'delta_y': -1})


def test_plot_bo():
    """Assert than plot_bo runs without errors."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'LDA'], n_calls=5, bo_params={'plot_bo': True})
    assert not atom.errors


def test_invalid_cv():
    """Assert than an error is raised when cv<=0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, 'LR', n_calls=5, bo_params={'cv': -1})


def test_invalid_early_stopping():
    """Assert than an error is raised when early_stopping<=0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(
        RuntimeError, atom.run, 'LGB', n_calls=5, bo_params={'early_stopping': -1})


def test_custom_dimensions():
    """Assert that the pipeline works with custom dimensions."""
    dim = [Integer(100, 1000, name='max_iter')]

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR', n_calls=5, bo_params={'dimensions': dim})
    assert list(atom.lr.best_params.keys()) == ['max_iter']


def test_all_base_estimators():
    """Assert that the pipeline works for all base estimators."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    for estimator in ('GP', 'ET', 'RF', 'GBRT'):
        atom.run('LR', n_calls=5, bo_params={'base_estimator': estimator})
        assert not atom.errors


def test_invalid_base_estimator():
    """Assert than an error is raised when the base_estimator is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError,
                  atom.run, 'LR', n_calls=5, bo_params={'base_estimator': 'unknown'})


def test_early_stopping():
    """Assert than early stopping works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['XGB', 'LGB', 'CatB'], n_calls=5, bo_params={'early_stopping': 0.1})
    for model in atom.models_:
        assert isinstance(model.evals, dict)


def test_verbose_is_1():
    """Assert that the pipeline works for verbose=1."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.run('LR', n_calls=5)
    assert atom.lr._pbar is not None


def test_bagging_attribute_types():
    """Assert that the bagging attributes have python types (not numpy)."""
    # For single-metric
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR', bagging=5)
    assert isinstance(atom.lr.score_bagging, list)
    assert isinstance(atom.lr.mean_bagging, float)

    # For multi-metric
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR', metric=('f1', 'auc', 'recall'), bagging=5)
    assert isinstance(atom.lr.score_bagging, list)
    assert isinstance(atom.lr.mean_bagging, list)


# Test prediction methods =================================================== >>

def test_invalid_method():
    """Assert that an error is raised when the model doesn't have the method."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('SGD')
    pytest.raises(AttributeError, atom.SGD.predict_proba, X_bin)


def test_transformations_first():
    """Assert that all transformations are applied before predicting."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode(max_onehot=None)
    atom.run('Tree')
    assert isinstance(atom.Tree.predict(X10_str), np.ndarray)


# Test properties =========================================================== >>

def test_all_prediction_properties():
    """Assert that all prediction properties are saved as attributes when called."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'SGD'])
    assert atom.lr.predict_train is atom.lr._predict_train
    assert atom.lr.predict_test is atom.lr._predict_test
    assert atom.lr.predict_proba_train is atom.lr._predict_proba_train
    assert atom.lr.predict_proba_test is atom.lr._predict_proba_test
    assert atom.lr.predict_log_proba_train is atom.lr._predict_log_proba_train
    assert atom.lr.predict_log_proba_test is atom.lr._predict_log_proba_test
    assert atom.sgd.decision_function_train is atom.sgd._decision_func_train
    assert atom.sgd.decision_function_test is atom.sgd._decision_func_test


def test_results_property():
    """Assert that the results property doesn't return columns with NaNs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr')
    assert 'mean_bagging' not in atom.lr.results


def test_dataset_property():
    """Assert that the dataset property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.dataset.equals(atom.mnb.dataset)
    assert check_scaling(atom.lr.dataset)


def test_train_property():
    """Assert that the train property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.train.equals(atom.mnb.train)
    assert check_scaling(atom.lr.train)


def test_test_property():
    """Assert that the test property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.train.equals(atom.mnb.train)
    assert check_scaling(atom.lr.train)


def test_X_property():
    """Assert that the X property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.X.equals(atom.mnb.X)
    assert check_scaling(atom.lr.X)


def test_y_property():
    """Assert that the y property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.y.equals(atom.mnb.y)
    assert atom.y.equals(atom.lr.y)


def test_X_train_property():
    """Assert that the X_train property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.X_train.equals(atom.mnb.X_train)
    assert check_scaling(atom.lr.X_train)


def test_X_test_property():
    """Assert that the X_test property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.X_test.equals(atom.mnb.X_test)
    assert check_scaling(atom.lr.X_test)


def test_y_train_property():
    """Assert that the y_train property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.y_train.equals(atom.mnb.y_train)
    assert atom.y_train.equals(atom.lr.y_train)


def test_y_test_property():
    """Assert that the y_test property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['MNB', 'LR'])
    assert atom.y_test.equals(atom.mnb.y_test)
    assert atom.y_test.equals(atom.lr.y_test)


def test_target_property():
    """Assert that the target property returns the original target."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('MNB')
    assert atom.target == atom.mnb.target


# Test utility methods ====================================================== >>

def test_calibrate_invalid_task():
    """Assert than an error is raised when task='regression'."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    pytest.raises(PermissionError, atom.calibrate)


def test_calibrate():
    """Assert that calibrate  method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('MNB')
    atom.calibrate(cv=3)
    assert isinstance(atom.mnb.model, CalibratedClassifierCV)

    # For cv='prefit'
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('MNB')
    atom.calibrate(cv='prefit')
    assert isinstance(atom.mnb.model, CalibratedClassifierCV)


def test_reset_predict_properties():
    """Assert that the prediction properties are reset after calibrating."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('MNB')
    atom.calibrate()
    assert not atom.mnb._predict_train


def test_scoring_metric_None():
    """Assert that the scoring methods works when metric is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('MNB')
    assert isinstance(atom.mnb.scoring(), str)


def test_scoring_metric():
    """Assert that the scoring methods works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('MNB')
    assert isinstance(atom.mnb.scoring('auc'), float)
    assert isinstance(atom.mnb.scoring('roc_auc'), float)


def test_scoring_custom_metrics():
    """Assert that the scoring methods works for custom metrics."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('MNB')
    for metric in ['cm', 'confusion_matrix']:
        assert isinstance(atom.mnb.scoring(metric), np.ndarray)
    for metric in ['tn', 'fp', 'fn', 'tp']:
        assert isinstance(atom.mnb.scoring(metric), int)
    for metric in ['lift', 'fpr', 'tpr', 'sup']:
        assert isinstance(atom.mnb.scoring(metric), float)


def test_invalid_metric():
    """Assert that invalid metrics return a string."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('OLS')
    assert isinstance(atom.ols.scoring('roc_auc'), str)


def test_save_model():
    """Assert that the save_model saves a pickle file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('MNB')
    atom.mnb.save_model(FILE_DIR + 'MNB_model')
    assert glob.glob(FILE_DIR + 'MNB_model.pkl')
