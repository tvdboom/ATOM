# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for modeloptimizer.py

"""

# Standard packages
import glob
import pytest
from sklearn.calibration import CalibratedClassifierCV
from skopt.learning import GaussianProcessRegressor
from skopt.callbacks import TimerCallback
from skopt.space.space import Integer

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.data_cleaning import Scaler
from .utils import FILE_DIR, X_bin, y_bin, X_reg, y_reg


# Test utilities =================================================== >>

def test_scaler():
    """Assert that a scaler is made for models that need scaling."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LGB", "LDA"])
    assert isinstance(atom.lgb.scaler, Scaler)
    assert atom.lda.scaler is None


def test_repr_method():
    """Assert that the __repr__ method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lda")
    assert len(str(atom.lda)) == 99


def test_results_property():
    """Assert that the results property doesn't return columns with NaNs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr")
    assert "mean_bagging" not in atom.lr.results


# Test bayesian_optimization ======================================= >>

def test_n_initial_points_lower_1():
    """Assert than an error is raised when n_initial_points<1."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"], n_calls=5, n_initial_points=(2, -1))
    assert atom.errors.get("LDA")


def test_n_calls_lower_n_initial_points():
    """Assert than an error is raised when n_calls<n_initial_points."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"], n_calls=(5, 2), n_initial_points=(2, 3))
    assert atom.errors.get("LDA")


def test_callbacks_class():
    """Assert that custom callbacks works as intended for a class."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_calls=5, bo_params={"callbacks": TimerCallback()})
    assert not atom.errors


def test_callbacks_sequence():
    """Assert that custom callbacks works as intended for a sequence."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_calls=5, bo_params={"callbacks": [TimerCallback()]})
    assert not atom.errors


def test_all_callbacks():
    """Assert that all callbacks pre-defined callbacks work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_calls=5, bo_params={"max_time": 50, "delta_x": 5, "delta_y": 5})
    assert not atom.errors


def test_invalid_max_time():
    """Assert than an error is raised when max_time<0."""
    kwargs = dict(models="LR", n_calls=5, bo_params={"max_time": -1})

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, **kwargs)


def test_invalid_delta_x():
    """Assert than an error is raised when delta_x<0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, "LR", n_calls=5, bo_params={"delta_x": -1})


def test_invalid_delta_y():
    """Assert than an error is raised when delta_y<0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, "LR", n_calls=5, bo_params={"delta_y": -1})


def test_plot_bo():
    """Assert than plot_bo runs without errors."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.1, n_jobs=-1, random_state=1)
    atom.run(
        models=["lSVM", "kSVM", "MLP"],
        n_calls=35,
        n_initial_points=20,
        bo_params={"plot_bo": True},
    )
    assert not atom.errors


def test_invalid_cv():
    """Assert than an error is raised when cv<=0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, "LR", n_calls=5, bo_params={"cv": -1})


def test_invalid_early_stopping():
    """Assert than an error is raised when early_stopping<=0."""
    kwargs = dict(models="LGB", n_calls=5, bo_params={"early_stopping": -1})

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, **kwargs)


def test_custom_dimensions():
    """Assert that the pipeline works with custom dimensions."""
    dim = [Integer(100, 1000, name="max_iter")]

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_calls=5, bo_params={"dimensions": dim})
    assert list(atom.lr.best_params.keys()) == ["max_iter"]


def test_all_base_estimators():
    """Assert that the pipeline works for all base estimators."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    for estimator in ("GP", "ET", "RF", "GBRT", GaussianProcessRegressor()):
        atom.run("LR", n_calls=5, bo_params={"base_estimator": estimator})
        assert not atom.errors


def test_estimator_kwargs():
    """Assert that the kwargs provided are correctly passed to the estimator."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_calls=5, bo_params={"acq_func": "EI"})
    assert not atom.errors


def test_invalid_base_estimator():
    """Assert than an error is raised when the base_estimator is invalid."""
    kwargs = dict(models="LR", n_calls=5, bo_params={"base_estimator": "unknown"})

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, **kwargs)


def test_sample_weights_fit():
    """Assert that sample weights can be used with the BO."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LGB",
        n_calls=5,
        est_params={"sample_weight_fit": list(range(len(atom.y_train)))},
    )
    assert not atom.errors


@pytest.mark.parametrize("model", ["XGB", "LGB", "CatB"])
def test_early_stopping(model):
    """Assert than early stopping works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(model, n_calls=5, bo_params={"early_stopping": 0.1, "cv": 1})
    assert isinstance(getattr(atom, model).evals, dict)


@pytest.mark.parametrize("model", ["XGB", "LGB", "CatB"])
def test_est_params_for_fit(model):
    """Assert that est_params is used for fit if ends in _fit."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(model, est_params={"early_stopping_rounds_fit": 20})
    assert getattr(atom, model)._stopped


def test_est_params_removed_from_bo():
    """Assert that all params in est_params are dropped from the BO."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", n_calls=5, est_params={"n_estimators": 220})
    assert "n_estimators" not in atom.lgb.bo.params[0]


def test_est_params_unknown_param():
    """Assert that an error is raised for an unknown parameter in est_params."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, "LGB", n_calls=5, est_params={"test": 220})


def test_verbose_is_1():
    """Assert that the pipeline works for verbose=1."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.run("LR", n_calls=5)
    assert atom.lr._pbar is not None


def test_bagging_is_negative():
    """Assert that an error is raised when bagging<0."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"], bagging=(2, -1))
    assert atom.errors.get("LDA")


def test_bagging_attribute_types():
    """Assert that the bagging attributes have python types (not numpy)."""
    # For single-metric
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", n_calls=5, bagging=5)
    assert isinstance(atom.lgb.metric_bagging, list)
    assert isinstance(atom.lgb.mean_bagging, float)

    # For multi-metric
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", metric=("f1", "auc", "recall"), bagging=5)
    assert isinstance(atom.lgb.metric_bagging[0], tuple)
    assert isinstance(atom.lgb.mean_bagging, list)


# Test utility methods ============================================= >>

def test_calibrate_invalid_task():
    """Assert than an error is raised when task="regression"."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("OLS")
    pytest.raises(PermissionError, atom.calibrate)


def test_calibrate():
    """Assert that calibrate method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.calibrate(cv=3)
    assert isinstance(atom.mnb.estimator, CalibratedClassifierCV)


def test_calibrate_prefit():
    """Assert that calibrate method works as intended when cv="prefit"."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.calibrate(cv="prefit")
    assert isinstance(atom.mnb.estimator, CalibratedClassifierCV)


def test_save_estimator():
    """Assert that the save_estimator saves a pickle file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.save_estimator(FILE_DIR + "auto")
    assert glob.glob(FILE_DIR + "MultinomialNB")
