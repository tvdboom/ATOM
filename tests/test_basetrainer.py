# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for basetrainer.py

"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import mlflow
import pytest
from mlflow.tracking.fluent import ActiveRun
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

from atom.training import DirectClassifier, DirectRegressor

from .conftest import (
    bin_test, bin_train, class_test, class_train, reg_test, reg_train,
)


# Test _prepare_parameters =========================================== >>

def test_all_classification_models():
    """Assert that the default value selects all models."""
    trainer = DirectClassifier(models=None, random_state=1)
    trainer.run(bin_train, bin_test)
    assert len(trainer.models) + len(trainer.errors) == 30


def test_all_regression_models():
    """Assert that the default value selects all models."""
    trainer = DirectRegressor(models=None, random_state=1)
    trainer.run(reg_train, reg_test)
    assert len(trainer.models) + len(trainer.errors) == 28


def test_model_is_predefined():
    """Assert that predefined models are accepted."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == "LR"


@patch.dict("sys.modules", {"lightgbm": None})
def test_package_not_installed():
    """Assert that an error is raised when the model's package is not installed."""
    trainer = DirectClassifier("LGB", random_state=1)
    with pytest.raises(ModuleNotFoundError, match=".*Unable to import.*"):
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
    with pytest.raises(ValueError, match=".*Unknown model.*"):
        trainer.run(bin_train, bin_test)


def test_multiple_same_models():
    """Assert that the same model can used with different names."""
    trainer = DirectClassifier(["lr", "lr2", "lr_3"], random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == ["LR", "LR2", "LR_3"]


def test_only_task_models():
    """Assert that an error is raised for models at invalid task."""
    trainer = DirectClassifier("OLS", random_state=1)  # Only regression
    with pytest.raises(ValueError, match=".*can't perform classification.*"):
        trainer.run(bin_train, bin_test)

    trainer = DirectRegressor("LDA", random_state=1)  # Only classification
    with pytest.raises(ValueError, match=".*can't perform regression.*"):
        trainer.run(reg_train, reg_test)


def test_reruns():
    """Assert that rerunning a trainer works."""
    trainer = DirectClassifier(["lr", "lda"], random_state=1)
    trainer.run(bin_train, bin_test)
    trainer.run(bin_train, bin_test)


def test_duplicate_models():
    """Assert that an error is raised with duplicate models."""
    trainer = DirectClassifier(["lr", "LR", "lgb"], random_state=1)
    with pytest.raises(ValueError, match=".*duplicate models.*"):
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


def test_metric_is_custom():
    """Assert that using the metric acronyms work."""
    trainer = DirectClassifier("LR", metric="fnr", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "false_negative_rate"


def test_metric_is_invalid_scorer_name():
    """Assert that an error is raised when scorer name is invalid."""
    trainer = DirectClassifier("LR", metric="test", random_state=1)
    with pytest.raises(ValueError, match=".*metric parameter.*"):
        trainer.run(bin_train, bin_test)


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


def test_sequence_parameters_invalid_length():
    """Assert that an error is raised when the length is invalid."""
    trainer = DirectClassifier("LR", n_trials=(2, 2), random_state=1)
    with pytest.raises(ValueError, match=".*length should be equal.*"):
        trainer.run(bin_train, bin_test)


def test_est_params_all_models():
    """Assert that est_params passes the parameters to all models."""
    trainer = DirectClassifier(
        models=["RF", "ET"],
        n_trials=1,
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


def test_est_params_for_fit():
    """Assert that est_params is used for fit if ends in _fit."""
    trainer = DirectClassifier(
        models="LGB",
        est_params={"n_estimators": 100, "early_stopping_rounds_fit": 2},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert len(trainer.lgb.evals[0]) < 100


def test_custom_distributions():
    """Assert that custom distributions can be defined."""
    trainer = DirectClassifier(
        models="LR",
        n_trials=1,
        ht_params={"distributions": {"max_iter": IntDistribution(10, 20)}},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert list(trainer.lr.best_params) == ["max_iter"]


def test_custom_distributions_is_all():
    """Assert that the custom distributions can be set for all models."""
    trainer = DirectClassifier(
        models=["LR1", "LR2"],
        n_trials=1,
        ht_params={
            "distributions": {
                "all": {"max_iter": IntDistribution(10, 20)},
                "LR2": {"penalty": CategoricalDistribution(["l1", "l2"])},
            },
        },
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert list(trainer.lr1.best_params) == ["max_iter"]
    assert list(trainer.lr2.best_params) == ["max_iter", "penalty"]


def test_custom_distributions_per_model():
    """Assert that the custom distributions are distributed over the models."""
    trainer = DirectClassifier(
        models=["LR1", "LR2"],
        n_trials=1,
        ht_params={
            "distributions": {
                "lr1": {"max_iter": IntDistribution(100, 200)},
                "lr2": {"max_iter": IntDistribution(300, 400)},
            },
        },
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert 100 <= trainer.lr1.best_params["max_iter"] <= 200
    assert 300 <= trainer.lr2.best_params["max_iter"] <= 400


def test_ht_params_kwargs():
    """Assert that kwargs are passed to the study or optimize method."""
    trainer = DirectClassifier(
        models="LR",
        n_trials=1,
        ht_params={"pruner": MedianPruner()},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.lr.study.pruner.__class__ == MedianPruner


# Test _core_iteration ============================================= >>

def test_sequence_parameters():
    """Assert that every model get his corresponding parameters."""
    trainer = DirectClassifier(
        models=["LR", "LGB"],
        n_trials=(2, 4),
        n_bootstrap=[2, 7],
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert len(trainer.LR.trials) == 2
    assert len(trainer.lgb.bootstrap) == 7


def test_mlflow_run_is_started():
    """Assert that a mlflow run starts with the run method."""
    trainer = DirectRegressor(models="OLS", experiment="test", random_state=1)
    trainer.run(reg_train, reg_test)
    assert isinstance(trainer.ols._run, ActiveRun)


def test_error_handling():
    """Assert that models with errors are removed from the pipeline."""
    trainer = DirectClassifier(
        models=["LR", "LDA"],
        n_trials=1,
        ht_params={"plot": True, "distributions": {"LDA": "test"}},
        experiment="test",
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.errors.get("LDA")
    assert "LDA" not in trainer.models
    assert "LDA" not in trainer.results.index
    assert len(plt.gcf().axes) == 0  # New figure -> plot has been closed
    assert mlflow.active_run() is None  # Run has been ended


def test_one_model_failed():
    """Assert that the model error is raised when it fails."""
    trainer = DirectClassifier(
        models="LR",
        n_trials=1,
        ht_params={"distributions": "test"},
        random_state=1,
    )
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_all_models_failed():
    """Assert that an error is raised when all models failed."""
    trainer = DirectClassifier(
        models=["LR", "RF"],
        n_trials=1,
        ht_params={"distributions": "test"},
        random_state=1,
    )
    with pytest.raises(RuntimeError, match=".*All models failed.*"):
        trainer.run(bin_train, bin_test)
