# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for models.py

"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from optuna.distributions import IntDistribution
from sklearn.ensemble import RandomForestRegressor
from optuna.pruners import PatientPruner
from atom import ATOMClassifier, ATOMRegressor
from atom.pipeline import Pipeline

from .conftest import X_bin, X_class, X_reg, y_bin, y_class, y_reg


@pytest.mark.parametrize("model", [RandomForestRegressor, RandomForestRegressor()])
def test_custom_models(model):
    """Assert that ATOM works with custom models."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=model, n_trials=1)
    assert atom.rfr._fullname == "RandomForestRegressor"


def test_all_models_binary():
    """Assert that all models work with binary classification."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.5, n_jobs=-1, random_state=1)
    atom.run(models=["!CatNB", "!RNN"], n_trials=5)
    assert not atom.errors
    assert "CatNB" not in atom.models


def test_all_models_multiclass():
    """Assert that all models work with multiclass classification."""
    atom = ATOMClassifier(X_class, y_class, n_rows=0.4, n_jobs=-1, random_state=1)
    atom.run(models=["!CatNB", "!KNN", "!RNN"], n_trials=5)
    assert not atom.errors
    assert "CatNB" not in atom.models


def test_all_models_regression():
    """Assert that all models work with regression."""
    atom = ATOMRegressor(X_reg, y_reg, n_rows=0.2, n_jobs=-1, random_state=1)
    atom.run(models=None, n_trials=5)
    assert not atom.errors


def test_models_sklearnex_classification():
    """Assert the sklearnex engine works for classification tasks."""
    atom = ATOMClassifier(X_bin, y_bin, device="cpu", engine="sklearnex", random_state=1)
    atom.run(models=["knn", "lr", "rf", "svm"], n_trials=1)
    assert not atom.errors


def test_models_sklearnex_regression():
    """Assert the sklearnex engine works for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, device="cpu", engine="sklearnex", random_state=1)
    atom.run(models=["en", "knn", "lasso", "ols", "rf", "ridge", "svm"], n_trials=1)
    assert not atom.errors


@patch.dict("sys.modules", {"cuml": MagicMock(spec=["__spec__"])})
def test_models_cuml_classification():
    """Assert that all classification models can be called with cuml."""
    atom = ATOMClassifier(X_bin, y_bin, device="gpu", engine="cuml", random_state=1)
    with pytest.raises(RuntimeError):
        atom.run(models=["!CatB", "!LGB", "!XGB"], n_trials=1)


@patch.dict("sys.modules", {"cuml": MagicMock(spec=["__spec__"])})
def test_models_cuml_regression():
    """Assert that all regression models can be called with cuml."""
    atom = ATOMRegressor(X_reg, y_reg, device="gpu", engine="cuml", random_state=1)
    with pytest.raises(RuntimeError):
        atom.run(models=["!CatB", "!LGB", "!XGB"], n_trials=1)


def test_CatNB():
    """Assert that the CatNB model works. Needs special dataset."""
    X = np.random.randint(5, size=(100, 100))
    y = np.random.randint(2, size=100)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CatNB", n_trials=1)
    assert not atom.errors
    assert hasattr(atom, "CatNB")


def test_RNN():
    """Assert that the RNN model works. Fails with default parameters."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("RNN", n_trials=1, est_params={"outlier_label": "most_frequent"})
    assert not atom.errors
    assert hasattr(atom, "RNN")


@pytest.mark.parametrize("model", ["CatB", "LGB", "XGB"])
def test_pruning_non_sklearn(model):
    """Assert that non-sklearn models can be pruned."""
    atom = ATOMClassifier(X_class, y_class, n_jobs=-1, random_state=1)
    atom.run(model, n_trials=2, ht_params={"pruner": PatientPruner(None, patience=2)})
    assert "PRUNED" in atom.winner.trials["state"].values


def test_MLP_custom_hidden_layer_sizes():
    """Assert that the MLP model can have custom hidden_layer_sizes."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom.run("MLP", n_trials=1, est_params={"hidden_layer_sizes": (31, 2)})
    assert "hidden_layer_1" not in atom.mlp.best_params
    assert atom.mlp.estimator.get_params()["hidden_layer_sizes"] == (31, 2)


def test_MLP_custom_n_layers():
    """Assert that the MLP model can have a custom number of hidden layers."""
    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
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


def test_stacking_feature_importance():
    """Assert that the feature_importance attr can be retrieved."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "RF"])
    atom.stacking()
    assert isinstance(atom.stack.feature_importance, pd.Series)


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


def test_voting_feature_importance():
    """Assert that the feature_importance attr can be retrieved."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.voting()
    assert isinstance(atom.vote.feature_importance, pd.Series)
