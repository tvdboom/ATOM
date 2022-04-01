# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for basetrainer.py

"""

from unittest.mock import patch

import pytest
from mlflow.tracking.fluent import ActiveRun
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from skopt.callbacks import TimerCallback
from skopt.space.space import Categorical, Integer

from atom.training import DirectClassifier, DirectRegressor
from atom.utils import CUSTOM_SCORERS, PlotCallback

from .utils import (
    bin_test, bin_train, class_test, class_train, mnist, reg_test, reg_train,
)


# Test _prepare_metric ============================================= >>

def test_invalid_sequence_parameter():
    """Assert that an error is raised for parameters with the wrong length."""
    trainer = DirectClassifier(
        models="LR",
        metric=f1_score,
        needs_proba=[True, False],
        random_state=1,
    )
    with pytest.raises(ValueError, match=r".*length should be equal.*"):
        trainer.run(bin_train, bin_test)


def test_metric_is_sklearn_scorer():
    """Assert that using a sklearn SCORER works."""
    trainer = DirectClassifier("LR", metric="balanced_accuracy", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "balanced_accuracy"


def test_metric_is_acronym():
    """Assert that using the metric acronyms work."""
    trainer = DirectClassifier("LR", metric="auc", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "roc_auc"


@pytest.mark.parametrize("metric", CUSTOM_SCORERS)
def test_metric_is_custom(metric):
    """Assert that using the metric acronyms work."""
    trainer = DirectClassifier("LR", metric=metric, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == CUSTOM_SCORERS[metric].__name__


def test_metric_is_invalid_scorer_name():
    """Assert that an error is raised when scorer name is invalid."""
    trainer = DirectClassifier("LR", metric="test", random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_metric_is_function():
    """Assert that a function metric works."""
    trainer = DirectClassifier("LR", metric=f1_score, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1_score"


def test_metric_is_scorer():
    """Assert that a scorer metric works."""
    trainer = DirectClassifier("LR", metric=make_scorer(f1_score), random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1"


# Test _prepare_parameters =========================================== >>

def test_all_classification_models():
    """Assert that the default value selects all models."""
    trainer = DirectClassifier(models=None, random_state=1)
    trainer.run(bin_train, bin_test)
    assert len(trainer.models) + len(trainer.errors) == 29


def test_all_regression_models():
    """Assert that the default value selects all models."""
    trainer = DirectRegressor(models=None, random_state=1)
    trainer.run(reg_train, reg_test)
    assert len(trainer.models) + len(trainer.errors) == 27


def test_multidim_predefined_model():
    """Assert that an error is raised for multidim datasets with predefined models."""
    trainer = DirectClassifier("OLS", random_state=1)
    with pytest.raises(ValueError, match=r".*Multidimensional datasets are.*"):
        trainer.run(*mnist)


def test_model_is_predefined():
    """Assert that predefined models are accepted."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == "LR"


@patch.dict("sys.modules", {"lightgbm": None})
def test_package_not_installed():
    """Assert that an error is raised when the model's package is not installed."""
    trainer = DirectClassifier("LGB", random_state=1)
    with pytest.raises(ModuleNotFoundError, match=r".*Unable to import.*"):
        trainer.run(bin_train, bin_test)


def test_model_is_custom():
    """Assert that custom models are accepted."""
    trainer = DirectClassifier(RandomForestClassifier, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == "RFC"


def test_models_get_right_name():
    """Assert that the model names are transformed to the correct acronyms."""
    trainer = DirectClassifier(["lR", "tReE"], random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == ["LR", "Tree"]


def test_invalid_model_name():
    """Assert that an error is raised when the model is unknown."""
    trainer = DirectClassifier(models="invalid", random_state=1)
    with pytest.raises(ValueError, match=r".*Unknown model.*"):
        trainer.run(bin_train, bin_test)


def test_multiple_same_models():
    """Assert that the same model can used with different names."""
    trainer = DirectClassifier(["lr", "lr2", "lr_3"], random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == ["LR", "LR2", "LR_3"]


def test_only_task_models():
    """Assert that an error is raised for models at invalid task."""
    trainer = DirectClassifier("OLS", random_state=1)  # Only regression
    with pytest.raises(ValueError, match=r".*can't perform classification.*"):
        trainer.run(bin_train, bin_test)

    trainer = DirectRegressor("LDA", random_state=1)  # Only classification
    with pytest.raises(ValueError, match=r".*can't perform regression.*"):
        trainer.run(reg_train, reg_test)


def test_reruns():
    """Assert that rerunning a trainer works."""
    trainer = DirectClassifier(["lr", "lda"], random_state=1)
    trainer.run(bin_train, bin_test)
    trainer.run(bin_train, bin_test)


def test_duplicate_models():
    """Assert that an error is raised with duplicate models."""
    trainer = DirectClassifier(["lr", "LR", "lgb"], random_state=1)
    with pytest.raises(ValueError, match=r".*duplicate models.*"):
        trainer.run(bin_train, bin_test)


def test_default_metric():
    """Assert that a default metric is assigned depending on the task."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1"

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(class_train, class_test)
    assert trainer.metric == "f1_weighted"

    trainer = DirectRegressor("LGB", random_state=1)
    trainer.run(reg_train, reg_test)
    assert trainer.metric == "r2"


def test_sequence_parameters_invalid_length():
    """Assert that an error is raised when the length is invalid."""
    trainer = DirectClassifier("LR", n_calls=(2, 2), random_state=1)
    with pytest.raises(ValueError, match=r".*Length should be equal.*"):
        trainer.run(bin_train, bin_test)


@pytest.mark.parametrize("n_calls", [-2, 1])
def test_n_calls_parameter_is_invalid(n_calls):
    """Assert that an error is raised when n_calls=1 or <0."""
    trainer = DirectClassifier("LR", n_calls=n_calls, random_state=1)
    with pytest.raises(ValueError, match=r".*n_calls parameter.*"):
        trainer.run(bin_train, bin_test)


def test_n_initial_points_parameter_is_zero():
    """Assert that an error is raised when n_initial_points=0."""
    trainer = DirectClassifier("LR", n_calls=2, n_initial_points=0, random_state=1)
    with pytest.raises(ValueError, match=r".*n_initial_points parameter.*"):
        trainer.run(bin_train, bin_test)


def test_n_bootstrap_parameter_is_below_zero():
    """Assert that an error is raised when n_bootstrap<0."""
    trainer = DirectClassifier("LR", n_bootstrap=-1, random_state=1)
    with pytest.raises(ValueError, match=r".*n_bootstrap parameter.*"):
        trainer.run(bin_train, bin_test)


def test_est_params_all_models():
    """Assert that est_params passes the parameters to all models."""
    trainer = DirectClassifier(
        models=["RF", "ET"],
        n_calls=2,
        n_initial_points=1,
        est_params={"n_estimators": 20, "all": {"bootstrap": False}},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.et.estimator.get_params()["n_estimators"] == 20
    assert trainer.rf.estimator.get_params()["bootstrap"] is False


def test_est_params_per_model():
    """Assert that est_params passes the parameters per model."""
    trainer = DirectClassifier(
        models=["XGB", "LGB"],
        est_params={"xgb": {"n_estimators": 15}, "lgb": {"n_estimators": 20}},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.xgb.estimator.get_params()["n_estimators"] == 15
    assert trainer.lgb.estimator.get_params()["n_estimators"] == 20


def test_est_params_default_method():
    """Assert that custom parameters overwrite the default ones."""
    trainer = DirectClassifier("RF", est_params={"n_jobs": 3}, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.rf.estimator.get_params()["n_jobs"] == 3
    assert trainer.rf.estimator.get_params()["random_state"] == 1


@pytest.mark.parametrize("model", ["XGB", "LGB", "CatB"])
def test_est_params_for_fit(model):
    """Assert that est_params is used for fit if ends in _fit."""
    trainer = DirectClassifier(
        models=model,
        est_params={"early_stopping_rounds_fit": 2},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert getattr(trainer, model)._stopped != ("---", "---")


def test_est_params_unknown_param():
    """Assert that unknown parameters in est_params are caught."""
    trainer = DirectClassifier(
        models=["LR", "LGB"],
        n_calls=5,
        est_params={"test": 220},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert list(trainer.errors.keys()) == ["LR"]  # LGB passes since it accepts kwargs


def test_est_params_unknown_param_fit():
    """Assert that unknown parameters in est_params_fit are caught."""
    trainer = DirectClassifier(
        models=["LR", "LGB"],
        est_params={"test_fit": 220},
        random_state=1,
    )
    with pytest.raises(RuntimeError):
        trainer.run(bin_train, bin_test)


def test_base_estimator_default():
    """Assert that GP is the default base estimator."""
    trainer = DirectClassifier("LR", n_calls=5, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer._bo["base_estimator"] == "GP"


def test_base_estimator_invalid():
    """Assert that an error is raised when the base estimator is invalid."""
    trainer = DirectClassifier("LR", bo_params={"base_estimator": "u"}, random_state=1)
    with pytest.raises(ValueError, match=r".*base_estimator parameter.*"):
        trainer.run(bin_train, bin_test)


@pytest.mark.parametrize("callback", [TimerCallback(), [TimerCallback()]])
def test_callback(callback):
    """Assert that custom callbacks are accepted."""
    trainer = DirectClassifier(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"callback": callback},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)


def test_all_callbacks():
    """Assert that all predefined callbacks work as intended."""
    trainer = DirectClassifier(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"max_time": 50, "delta_x": 5, "delta_y": 5},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)


def test_invalid_max_time():
    """Assert than an error is raised when max_time<=0."""
    trainer = DirectClassifier("LR", bo_params={"max_time": 0}, random_state=1)
    with pytest.raises(ValueError, match=r".*max_time parameter.*"):
        trainer.run(bin_train, bin_test)


def test_invalid_delta_x():
    """Assert than an error is raised when delta_x<0."""
    trainer = DirectClassifier("LR", bo_params={"delta_x": -2}, random_state=1)
    with pytest.raises(ValueError, match=r".*delta_x parameter.*"):
        trainer.run(bin_train, bin_test)


def test_invalid_delta_y():
    """Assert than an error is raised when delta_y<0."""
    trainer = DirectClassifier("LR", bo_params={"delta_y": -2}, random_state=1)
    with pytest.raises(ValueError, match=r".*delta_y parameter.*"):
        trainer.run(bin_train, bin_test)


def test_plot():
    """Assert that plotting the BO runs without errors."""
    trainer = DirectClassifier(
        models=["lSVM", "kSVM", "MLP"],
        n_calls=(17, 17, 40),
        n_initial_points=8,
        bo_params={"plot": True},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)


def test_invalid_cv():
    """Assert than an error is raised when cv<=0."""
    trainer = DirectClassifier("LR", bo_params={"cv": 0}, random_state=1)
    with pytest.raises(ValueError, match=r".*cv parameter.*"):
        trainer.run(bin_train, bin_test)


def test_invalid_early_stopping():
    """Assert than an error is raised when early_stopping<=0."""
    trainer = DirectClassifier("LR", bo_params={"early_stopping": -1}, random_state=1)
    with pytest.raises(ValueError, match=r".*early_stopping parameter.*"):
        trainer.run(bin_train, bin_test)


def test_custom_dimensions_is_list():
    """Assert that the custom dimensions are for all models if list."""
    trainer = DirectClassifier(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"dimensions": [Integer(10, 20, name="max_iter")]},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert list(trainer.lr.best_params) == ["max_iter"]


def test_custom_dimensions_is_all():
    """Assert that the custom dimensions can be set for all models."""
    trainer = DirectClassifier(
        models=["LR1", "LR2"],
        n_calls=2,
        n_initial_points=2,
        bo_params={
            "dimensions": {
                "all": [Integer(10, 20, name="max_iter")],
                "LR2": Categorical(["l1", "l2"], name="penalty"),
            },
        },
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert list(trainer.lr1.best_params) == ["max_iter"]
    assert list(trainer.lr2.best_params) == ["max_iter", "penalty"]


def test_custom_dimensions_per_model():
    """Assert that the custom dimensions are distributed over the models."""
    trainer = DirectClassifier(
        models=["LR1", "LR2"],
        n_calls=2,
        n_initial_points=2,
        bo_params={
            "dimensions": {
                "lr1": [Integer(100, 200, name="max_iter")],
                "lr2": [Integer(300, 400, name="max_iter")],
            },
        },
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert 100 <= trainer.lr1.best_params["max_iter"] <= 200
    assert 300 <= trainer.lr2.best_params["max_iter"] <= 400


def test_optimizer_kwargs():
    """Assert that the kwargs provided are passed to the optimizer."""
    trainer = DirectClassifier(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"acq_func": "EI"},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer._bo["kwargs"].get("acq_func") == "EI"


# Test _core_iteration ============================================= >>

def test_sequence_parameters():
    """Assert that every model get his corresponding parameters."""
    trainer = DirectClassifier(
        models=["LR", "Tree", "LGB"],
        n_calls=(2, 3, 4),
        n_initial_points=(1, 2, 3),
        n_bootstrap=[2, 5, 7],
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert len(trainer.LR.bo) == 2
    assert sum(trainer.tree.bo["call"].str.startswith("Initial")) == 2
    assert len(trainer.lgb.metric_bootstrap) == 7


def test_custom_dimensions_for_bo():
    """Assert that the BO runs when custom dimensions are provided."""
    trainer = DirectRegressor(
        models="OLS",
        n_calls=5,
        bo_params={"dimensions": [Categorical([True, False], name="fit_intercept")]},
        random_state=1,
    )
    trainer.run(reg_train, reg_test)
    assert not trainer.ols.bo.empty


def test_mlflow_run_is_started():
    """Assert that a mlflow run starts with the run method."""
    trainer = DirectRegressor(models="OLS", experiment="test", random_state=1)
    trainer.run(reg_train, reg_test)
    assert isinstance(trainer.ols._run, ActiveRun)


def test_error_handling():
    """Assert that models with errors are removed from the pipeline."""
    trainer = DirectClassifier(
        models=["LR", "LDA"],
        n_calls=4,
        n_initial_points=[2, 5],
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.errors.get("LDA")
    assert "LDA" not in trainer.models
    assert "LDA" not in trainer.results.index


def test_close_plot_after_error():
    """Assert that the BO plot is closed after an error."""
    trainer = DirectClassifier(
        models=["LR", "LDA"],
        n_calls=4,
        n_initial_points=[2, 5],
        bo_params={"plot": True},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert PlotCallback.c == 1  # First model is 0, after error passes to 1


def test_one_model_failed():
    """Assert that the model error is raised when it fails."""
    trainer = DirectClassifier("LR", n_calls=4, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_all_models_failed():
    """Assert that an error is raised when all models failed."""
    trainer = DirectClassifier(["LR", "RF"], n_calls=4, random_state=1)
    pytest.raises(RuntimeError, trainer.run, bin_train, bin_test)
