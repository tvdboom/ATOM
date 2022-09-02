# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for models.py

"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from skopt.space.space import Integer

from atom import ATOMClassifier, ATOMRegressor
from atom.feature_engineering import FeatureSelector
from atom.models import MODELS
from atom.pipeline import Pipeline

from .conftest import X_bin, X_class2, X_reg, y_bin, y_class2, y_reg


binary, multiclass, regression = [], [], []
for m in MODELS.values():
    if "class" in m._estimators:
        if m.acronym != "CatNB":
            binary.append(m.acronym)  # CatNB needs a special dataset
        if not m.acronym.startswith("Cat"):
            multiclass.append(m.acronym)  # CatB fails with error on their side
    if "reg" in m._estimators:
        regression.append(m.acronym)


@pytest.mark.parametrize("model", [RandomForestRegressor, RandomForestRegressor()])
def test_custom_models(model):
    """Assert that ATOM works with custom models."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=model, n_calls=2, n_initial_points=1)
    assert atom.rfr.fullname == "RandomForestRegressor"
    assert atom.rfr.estimator.get_params()["random_state"] == 1


@pytest.mark.parametrize("model", binary)
def test_models_binary(model):
    """Assert that all models work with binary classification."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.24, random_state=1)
    atom.run(
        models=model,
        metric="auc",
        n_calls=2,
        n_initial_points=1,
        bo_params={"base_estimator": "rf", "cv": 1},
    )
    assert not atom.errors
    assert hasattr(atom, model)


@pytest.mark.parametrize("model", multiclass)
def test_models_multiclass(model):
    """Assert that all models work with multiclass classification."""
    atom = ATOMClassifier(X_class2, y_class2, test_size=0.24, random_state=1)
    atom.run(
        models=model,
        metric="f1_micro",
        n_calls=2,
        n_initial_points=1,
        bo_params={"base_estimator": "rf", "cv": 1},
    )
    assert not atom.errors
    assert hasattr(atom, model)


@pytest.mark.parametrize("model", regression)
def test_models_regression(model):
    """Assert that all models work with regression."""
    atom = ATOMRegressor(X_reg, y_reg, test_size=0.24, random_state=1)
    atom.run(
        models=model,
        metric="neg_mean_absolute_error",
        n_calls=2,
        n_initial_points=1,
        bo_params={"base_estimator": "gbrt", "cv": 1},
    )
    assert not atom.errors
    assert hasattr(atom, model)


def test_Dummy():
    """Assert that Dummy doesn't crash when strategy=quantile."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(
        models="dummy",
        n_calls=2,
        n_initial_points=1,
        est_params={"strategy": "quantile"},
    )


def test_CatNB():
    """Assert that the CatNB model works. Separated because of special dataset."""
    X = np.random.randint(5, size=(100, 100))
    y = np.random.randint(2, size=100)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CatNB", n_calls=2, n_initial_points=1)
    assert not atom.errors
    assert hasattr(atom, "CatNB")


def test_LR():
    """Assert that elasticnet doesn't crash with default l1_ratio."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LR",
        n_calls=2,
        n_initial_points=1,
        est_params={"penalty": "elasticnet", "solver": "saga"},
    )
    assert atom.lr.bo["params"][0]["l1_ratio"] is not None


def test_RNN():
    """Assert that the RNN model works when called just for the estimator."""
    fs = FeatureSelector("rfe", solver="RNN_class", n_features=10)
    with pytest.raises(ValueError):
        # Fails cause RNN has no coef_ nor feature_importances_ attribute
        fs.fit_transform(X_bin, y_bin)


def test_MLP_custom_hidden_layer_sizes():
    """Assert that the MLP model can have custom hidden_layer_sizes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="MLP",
        n_calls=2,
        n_initial_points=1,
        est_params={"hidden_layer_sizes": (31, 2)},
    )
    assert atom.mlp.estimator.get_params()["hidden_layer_sizes"] == (31, 2)


def test_MLP_custom_n_layers():
    """Assert that the MLP model can have a custom number of hidden layers."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="MLP",
        n_calls=2,
        n_initial_points=1,
        bo_params={
            "dimensions": [
                Integer(0, 100, name="hidden_layer_1"),
                Integer(0, 20, name="hidden_layer_2"),
                Integer(0, 20, name="hidden_layer_3"),
                Integer(0, 20, name="hidden_layer_4"),
            ]
        },
    )
    assert atom.mlp.bo["params"][0]["hidden_layer_sizes"] == (100,)


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
    with pytest.raises(ValueError, match=r".*on the current branch.*"):
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
    with pytest.raises(ValueError, match=r".*on the current branch.*"):
        atom.voting(models=["LR", "LDA"])
