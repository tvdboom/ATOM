# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for training.py

"""

# Standard packages
import pytest
import pandas as pd

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


# Test Trainer ===================================================== >>

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


# Test SuccessiveHalving =========================================== >>

def test_skip_runs_below_zero():
    """Assert that an error is raised if skip_runs < 0."""
    sh = SuccessiveHalvingRegressor(models="OLS", skip_runs=-1)
    pytest.raises(ValueError, sh.run, reg_train, reg_test)


def test_skip_runs_too_large():
    """Assert that an error is raised if skip_runs >= n_runs."""
    sh = SuccessiveHalvingRegressor(models=["OLS", "BR"], skip_runs=2)
    pytest.raises(ValueError, sh.run, reg_train, reg_test)


def test_successive_halving_results_is_multi_index():
    """Assert that the results property is a multi-index dataframe."""
    sh = SuccessiveHalvingRegressor(["Tree", "RF", "AdaB", "LGB"], random_state=1)
    sh.run(reg_train, reg_test)
    assert len(sh.results) == 7  # 4 + 2 + 1
    assert isinstance(sh.results.index, pd.MultiIndex)
    assert sh.results.index.names == ["n_models", "model"]


def test_models_are_reset():
    """Assert that the models attributes is reset after fitting."""
    sh = SuccessiveHalvingRegressor(["Tree", "RF", "AdaB", "LGB"], random_state=1)
    sh.run(reg_train, reg_test)
    assert sh.models == ["Tree", "RF", "AdaB", "LGB"]


def test_successive_halving_train_index_is_reset():
    """Assert that the train index is reset after fitting."""
    sh = SuccessiveHalvingRegressor(["Tree", "RF", "AdaB", "LGB"], random_state=1)
    sh.run(reg_train, reg_test)
    assert sh.branch.idx[0] == len(reg_train)


# Test TrainSizing ================================================= >>

def test_train_sizing_results_is_multi_index():
    """Assert that the results property is a multi-index dataframe."""
    ts = TrainSizingRegressor(["RF", "LGB"], train_sizes=[100, 200], random_state=1)
    ts.run(reg_train, reg_test)
    assert len(ts.results) == 4  # 2 models * 2 runs
    assert isinstance(ts.results.index, pd.MultiIndex)
    assert ts.results.index.names == ["frac", "model"]


def test_train_sizing_train_index_is_reset():
    """Assert that the train index is reset after fitting."""
    ts = TrainSizingRegressor("LGB", train_sizes=[0.6, 0.8], random_state=1)
    ts.run(reg_train, reg_test)
    assert ts.branch.idx[0] == len(reg_train)


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
