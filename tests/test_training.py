"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for training.py

"""

import pytest

from atom.training import (
    DirectClassifier, DirectForecaster, DirectRegressor,
    SuccessiveHalvingClassifier, SuccessiveHalvingForecaster,
    SuccessiveHalvingRegressor, TrainSizingClassifier, TrainSizingForecaster,
    TrainSizingRegressor,
)

from .conftest import reg_test, reg_train


# Test trainers ============================================== >>

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
    assert "Tree" not in sh._models  # The original model is deleted
    assert all(m in sh.models for m in ("Tree4", "AdaB2", "AdaB1"))


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
    assert trainer._goal.name == "classification"

    trainer = DirectForecaster("NF")
    assert trainer._goal.name == "forecast"

    trainer = DirectRegressor("OLS")
    assert trainer._goal.name == "regression"


def test_goals_successive_halving():
    """Assert that the goal of every SuccessiveHalving class is set correctly."""
    trainer = SuccessiveHalvingClassifier("LR")
    assert trainer._goal.name == "classification"

    trainer = SuccessiveHalvingForecaster("NF")
    assert trainer._goal.name == "forecast"

    trainer = SuccessiveHalvingRegressor("OLS")
    assert trainer._goal.name == "regression"


def test_goals_train_sizing():
    """Assert that the goal of every TrainSizing class is set correctly."""
    trainer = TrainSizingClassifier("LR")
    assert trainer._goal.name == "classification"

    trainer = TrainSizingForecaster("NF")
    assert trainer._goal.name == "forecast"

    trainer = TrainSizingRegressor("OLS")
    assert trainer._goal.name == "regression"
