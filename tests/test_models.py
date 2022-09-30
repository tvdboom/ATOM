# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for models.py

"""

import numpy as np
import pytest
from optuna.distributions import IntDistribution
from sklearn.ensemble import RandomForestRegressor

from atom import ATOMClassifier, ATOMRegressor
from atom.models import MODELS
from atom.pipeline import Pipeline

from .conftest import X_bin, X_class, X_reg, y_bin, y_class, y_reg


binary, multiclass, regression = [], [], []
for m in MODELS.values():
    if "class" in m._estimators and m.acronym != "CatNB":
        # CatNB needs a special dataset
        binary.append(m.acronym)
        if m.acronym != "RNN":  # RNN fails with sklearn's default parameters
            multiclass.append(m.acronym)
    if "reg" in m._estimators:
        regression.append(m.acronym)


@pytest.mark.parametrize("model", [RandomForestRegressor, RandomForestRegressor()])
def test_custom_models(model):
    """Assert that ATOM works with custom models."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=model, n_trials=1)
    assert atom.rfr._fullname == "RandomForestRegressor"


@pytest.mark.parametrize("model", binary)
def test_models_binary(model):
    """Assert that all models work with binary classification."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.2, n_jobs=-1, random_state=1)
    atom.run(models=model, n_trials=1)
    assert not atom.errors
    assert hasattr(atom, model)


@pytest.mark.parametrize("model", multiclass)
def test_models_multiclass(model):
    """Assert that all models work with multiclass classification."""
    atom = ATOMClassifier(X_class, y_class, n_rows=0.4, n_jobs=-1, random_state=1)
    atom.run(models=model, n_trials=1)
    assert not atom.errors
    assert hasattr(atom, model)


@pytest.mark.parametrize("model", regression)
def test_models_regression(model):
    """Assert that all models work with regression."""
    atom = ATOMRegressor(X_reg, y_reg, n_rows=0.2, n_jobs=-1, random_state=1)
    atom.run(models=model, n_trials=1)
    assert not atom.errors
    assert hasattr(atom, model)


def test_CatNB():
    """Assert that the CatNB model works. Needs special dataset."""
    X = np.random.randint(5, size=(100, 100))
    y = np.random.randint(2, size=100)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CatNB", n_trials=1)
    assert not atom.errors
    assert hasattr(atom, "CatNB")


def test_RNN():
    """Assert that the RNN model works. Fails with sklearn's parameters."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("RNN", n_trials=1, est_params={"outlier_label": "most_frequent"})
    assert not atom.errors
    assert hasattr(atom, "RNN")


def test_MLP_custom_hidden_layer_sizes():
    """Assert that the MLP model can have custom hidden_layer_sizes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MLP", n_trials=1, est_params={"hidden_layer_sizes": (31, 2)})
    assert "hidden_layer_1" not in atom.mlp.best_params
    assert atom.mlp.estimator.get_params()["hidden_layer_sizes"] == (31, 2)


def test_MLP_custom_n_layers():
    """Assert that the MLP model can have a custom number of hidden layers."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="MLP",
        n_trials=1,
        ht_params={
            "distributions": {
                "hidden_layer_1": IntDistribution(2, 4),
                "hidden_layer_2": IntDistribution(2, 4),
                "hidden_layer_3": IntDistribution(2, 4),
                "hidden_layer_4": IntDistribution(2, 4),
            }
        },
    )
    assert len(atom.mlp.trials["params"][0]["hidden_layer_sizes"]) == 4


# Test ensembles =================================================== >>

def test_stacking():
    """Assert that the Stacking model works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=["OLS", "RF"])
    atom.stacking()
    assert isinstance(atom.stack.estimator.estimators_[0], Pipeline)
    assert isinstance(atom.stack.estimator.estimators_[1], RandomForestRegressor)


def test_stacking_multiple_branches():
    """Assert that an error is raised when branches are different."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    atom.branch = "2"
    atom.run("LDA")
    with pytest.raises(ValueError, match=".*on the current branch.*"):
        atom.stacking(models=["LR", "LDA"])


def test_voting():
    """Assert that the Voting model works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=["OLS", "RF"])
    atom.voting()
    assert isinstance(atom.vote.estimator.estimators_[0], Pipeline)
    assert isinstance(atom.vote.estimator.estimators_[1], RandomForestRegressor)


def test_voting_multiple_branches():
    """Assert that an error is raised when branches are different."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.branch = "2"
    with pytest.raises(ValueError, match=".*on the current branch.*"):
        atom.voting(models=["LR", "LDA"])
