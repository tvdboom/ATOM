# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for training.py

"""

# Import packages
import pytest
import pandas as pd

# Own modules
from atom.training import (
    TrainerClassifier, TrainerRegressor,
    SuccessiveHalvingClassifier, SuccessiveHalvingRegressor,
    TrainSizingClassifier, TrainSizingRegressor
)
from .utils import bin_train, bin_test, class_train, class_test, reg_train, reg_test


# Test Trainer ============================================================== >>

def test_infer_task():
    """Assert that the correct task is inferred from the data."""
    trainer = TrainerClassifier('LR')
    trainer.run(bin_train, bin_test)
    assert trainer.task == 'binary classification'

    trainer = TrainerClassifier('LR')
    trainer.run(class_train, class_test)
    assert trainer.task == 'multiclass classification'

    trainer = TrainerRegressor('OLS')
    trainer.run(reg_train, reg_test)
    assert trainer.task == 'regression'


def test_default_metric():
    """Assert that a default metric_ is assigned depending on the task."""
    trainer = TrainerClassifier('LR')
    trainer.run(bin_train, bin_test)
    assert trainer.metric == 'f1'

    trainer = TrainerClassifier('LR')
    trainer.run(class_train, class_test)
    assert trainer.metric == 'f1_weighted'

    trainer = TrainerRegressor('OLS')
    trainer.run(reg_train, reg_test)
    assert trainer.metric == 'r2'


# Test SuccessiveHalving ==================================================== >>

def test_invalid_skip_iter():
    """Assert that an error is raised if skip_iter < 0."""
    pytest.raises(ValueError, SuccessiveHalvingRegressor, 'OLS', skip_iter=-1)


def test_successive_halving_results_is_multi_index():
    """Assert that the results property is a multi-index dataframe."""
    sh = SuccessiveHalvingRegressor(['OLS', 'BR', 'RF', 'LGB'])
    sh.run(reg_train, reg_test)
    assert len(sh.results) == 7  # 4 + 2 + 1
    assert isinstance(sh.results.index, pd.MultiIndex)
    assert sh.results.index.names == ['run', 'model']


def test_models_are_reset():
    """Assert that the models attributes is reset after fitting."""
    sh = SuccessiveHalvingRegressor(['OLS', 'BR', 'RF', 'LGB'])
    sh.run(reg_train, reg_test)
    assert sh.models == ['OLS', 'BR', 'RF', 'LGB']


def test_successive_halving_train_index_is_reset():
    """Assert that the train index is reset after fitting."""
    sh = SuccessiveHalvingRegressor(['OLS', 'BR', 'RF', 'LGB'])
    sh.run(reg_train, reg_test)
    assert sh._idx[0] == len(reg_train)


# Test TrainSizing ========================================================== >>

def test_train_sizing_results_is_multi_index():
    """Assert that the results property is a multi-index dataframe."""
    ts = TrainSizingRegressor(['OLS', 'BR'])
    ts.run(reg_train, reg_test)
    assert len(ts.results) == 10  # 2 models * 5 runs
    assert isinstance(ts.results.index, pd.MultiIndex)
    assert ts.results.index.names == ['run', 'model']


def test_sizes_attribute():
    """Assert that the _sizes attributes is reset after fitting."""
    ts = TrainSizingRegressor('OLS', train_sizes=(0.1, 100))
    ts.run(reg_train, reg_test)
    assert ts._sizes == [int(0.1 * len(reg_train)), 100]


def test_train_sizing_train_index_is_reset():
    """Assert that the train index is reset after fitting."""
    ts = TrainSizingRegressor('OLS')
    ts.run(reg_train, reg_test)
    assert ts._idx[0] == len(reg_train)


# Test goals ================================================================ >>

def test_goals_trainers():
    """Assert that the goal of every Trainer class is set correctly."""
    trainer = TrainerClassifier('LR')
    assert trainer.goal == 'classification'

    trainer = TrainerRegressor('OLS')
    assert trainer.goal == 'regression'


def test_goals_successive_halving():
    """Assert that the goal of every SuccessiveHalving class is set correctly."""
    sh = SuccessiveHalvingClassifier('LR')
    assert sh.goal == 'classification'

    sh = SuccessiveHalvingRegressor('OLS')
    assert sh.goal == 'regression'


def test_goals_train_sizing():
    """Assert that the goal of every TrainSizing class is set correctly."""
    ts = TrainSizingClassifier('LR')
    assert ts.goal == 'classification'

    ts = TrainSizingRegressor('OLS')
    assert ts.goal == 'regression'
