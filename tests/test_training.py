# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for training.py

"""

import pytest

from atom.training import (
    DirectClassifier, DirectRegressor, SuccessiveHalvingClassifier,
    SuccessiveHalvingRegressor, TrainSizingClassifier, TrainSizingRegressor,
)

from .conftest import (
    bin_test, bin_train, class_test, class_train, reg_test, reg_train,
)


# Test trainers ============================================== >>

def test_infer_task():
    """Assert that the correct task is inferred from the data."""
    trainer = DirectClassifier("LR")
    trainer.run(bin_train, bin_test)
    assert trainer.task == "binary classification"

    trainer = DirectClassifier("LR")
    trainer.run(class_train, class_test)
    assert trainer.task == "multiclass classification"

    trainer = DirectRegressor("LGB", est_params={"n_estimators": 5})
    trainer.run(reg_train, reg_test)
    assert trainer.task == "regression"


def test_sh_skip_runs_below_zero():
    """Assert that an error is raised if skip_runs < 0."""
    sh = SuccessiveHalvingRegressor(models="OLS", skip_runs=-1)
    pytest.raises(ValueError, sh.run, reg_train, reg_test)


def test_sh_skip_runs_too_large():
    """Assert that an error is raised if skip_runs >= n_runs."""
    sh = SuccessiveHalvingRegressor(models=["OLS", "BR"], skip_runs=2)
    pytest.raises(ValueError, sh.run, reg_train, reg_test)


def test_models_are_restored():
    """Assert that the models attributes are all restored after fitting."""
    sh = SuccessiveHalvingRegressor(
        models=["Tree", "AdaB", "RF", "LGB"],
        est_params={
            "AdaB": {"n_estimators": 5},
            "RF": {"n_estimators": 5},
            "LGB": {"n_estimators": 5},
        },
        random_state=1,
    )
    sh.run(reg_train, reg_test)
    assert "Tree" not in sh._models  # Original model is deleted
    assert all(m in sh.models for m in ("Tree4", "AdaB2", "LGB1"))


def test_ts_int_train_sizes():
    """Assert that train sizing accepts different types as sizes."""
    sh = TrainSizingClassifier("Tree", train_sizes=5, random_state=1)
    sh.run(reg_train, reg_test)
    assert len(sh.tree02.train) == 61
    assert len(sh.tree06.train) == 185


def test_ts_different_train_sizes_types():
    """Assert that train sizing accepts different types as sizes."""
    sh = TrainSizingClassifier("Tree", train_sizes=[0.2, 200], random_state=1)
    sh.run(reg_train, reg_test)
    assert len(sh.tree02.train) == 61
    assert len(sh.tree065.train) == 200


# Test goals ======================================================= >>

def test_goals_trainers():
    """Assert that the goal of every Trainer class is set correctly."""
    trainer = DirectClassifier("LR")
    assert trainer.goal == "class"

    trainer = DirectRegressor("OLS")
    assert trainer.goal == "reg"


def test_goals_successive_halving():
    """Assert that the goal of every SuccessiveHalving class is set correctly."""
    sh = SuccessiveHalvingClassifier("LR")
    assert sh.goal == "class"

    sh = SuccessiveHalvingRegressor("OLS")
    assert sh.goal == "reg"


def test_goals_train_sizing():
    """Assert that the goal of every TrainSizing class is set correctly."""
    ts = TrainSizingClassifier("LR")
    assert ts.goal == "class"

    ts = TrainSizingRegressor("OLS")
    assert ts.goal == "reg"
