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

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from .utils import FILE_DIR, X_bin, y_bin, X_reg, y_reg


# Test utilities =================================================== >>

def test_scaler():
    """Assert that a scaler is made for models that need scaling."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LGB", "LDA"])
    assert atom.lgb.scaler and not atom.lda.scaler


def test_repr_method():
    """Assert that the __repr__ method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LDA")
    assert str(atom.lda).startswith("Linear Discriminant")


# Test bayesian_optimization ======================================= >>

def test_n_calls_lower_n_initial_points():
    """Assert than an error is raised when n_calls<n_initial_points."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"], n_calls=(5, 2), n_initial_points=(2, 3))
    assert atom.errors.get("LDA")


def test_est_params_removed_from_bo():
    """Assert that all params in est_params are dropped from the BO."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", n_calls=5, est_params={"n_estimators": 220})
    assert "n_estimators" not in atom.lgb.bo.params[0]


def test_est_params_unknown_param():
    """Assert that unknown parameters in est_params are caught."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], n_calls=5, est_params={"test": 220})
    assert list(atom.errors.keys()) == ["LR"]  # LGB passes since it accepts kwargs


def test_est_params_unknown_param_fit():
    """Assert that unknown parameters in est_params_fit are caught."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, ["LR", "LGB"], est_params={"test_fit": 220})


def test_est_params_default_method():
    """Assert that custom parameters overwrite the default ones."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("RF", est_params={"n_jobs": 3})
    assert atom.rf.estimator.get_params()["n_jobs"] == 3
    assert atom.rf.estimator.get_params()["random_state"] == 1


@pytest.mark.parametrize("est", ["GP", "ET", "RF", "GBRT", GaussianProcessRegressor()])
def test_all_base_estimators(est):
    """Assert that the pipeline works for all base estimators."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_calls=5, bo_params={"base_estimator": est})


def test_sample_weights_fit():
    """Assert that sample weights can be used with the BO."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LGB",
        n_calls=5,
        est_params={"sample_weight_fit": list(range(len(atom.y_train)))},
    )


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


def test_verbose_is_1():
    """Assert that the pipeline works for verbose=1."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=1, random_state=1)
    atom.run("LR", n_calls=5)
    assert atom.lr._pbar is not None


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


def test_calibrate_reset_predictions():
    """Assert that the prediction attributes are reset after calibrating."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    print(atom.mnb.score_test)
    atom.calibrate()
    assert atom.mnb._pred_attrs[9] is None


def test_rename():
    """Assert that the model's tag can be changed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.rename("_2")
    assert atom.models == "MNB_2"
    atom.mnb_2.rename("mnb_3")
    assert atom.models == "MNB_3"


def test_save_estimator():
    """Assert that the save_estimator saves a pickle file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.save_estimator(FILE_DIR + "auto")
    assert glob.glob(FILE_DIR + "MultinomialNB")
