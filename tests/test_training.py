# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for training.py

"""

# Standard packages
import pytest

# Own modules
from atom.training import (
    DirectClassifier,
    DirectRegressor,
    SuccessiveHalvingClassifier,
    SuccessiveHalvingRegressor,
    TrainSizingClassifier,
    TrainSizingRegressor,
)
from .utils import bin_train, bin_test, class_train, class_test, reg_train, reg_test


# Test trainers ============================================== >>

def test_infer_task():
    """Assert that the correct task is inferred from the data."""
    trainer = DirectClassifier("LR")
    trainer.run(bin_train, bin_test)
    assert trainer.task == "binary classification"

    trainer = DirectClassifier("LR")
    trainer.run(class_train, class_test)
    assert trainer.task == "multiclass classification"

    trainer = DirectRegressor("LGB")
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
    sh = SuccessiveHalvingRegressor(["Tree", "RF", "AdaB", "LGB"], random_state=1)
    sh.run(reg_train, reg_test)
    assert "Tree" not in sh._models  # Original model is deleted
    assert all(m in sh.models for m in ("Tree4", "RF2", "AdaB1"))


def test_ts_different_train_sizes_types():
    """Assert that train sizing accepts different types as sizes."""
    sh = TrainSizingClassifier("Tree", train_sizes=[0.2, 200], random_state=1)
    sh.run(reg_train, reg_test)
    assert len(sh.tree02.train) == 61
    assert len(sh.tree0647.train) == 200


# Test goals ======================================================= >>

def test_goals_trainers():
    """Assert that the goal of every Trainer class is set correctly."""
    trainer = DirectClassifier("LR")
    assert trainer.goal == "classification"

    trainer = DirectRegressor("OLS")
    assert trainer.goal == "regression"


def test_goals_successive_halving():
    """Assert that the goal of every SuccessiveHalving class is set correctly."""
    sh = SuccessiveHalvingClassifier("LR")
    assert sh.goal == "classification"

    sh = SuccessiveHalvingRegressor("OLS")
    assert sh.goal == "regression"


def test_goals_train_sizing():
    """Assert that the goal of every TrainSizing class is set correctly."""
    ts = TrainSizingClassifier("LR")
    assert ts.goal == "classification"

    ts = TrainSizingRegressor("OLS")
    assert ts.goal == "regression"
