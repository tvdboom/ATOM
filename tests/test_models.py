# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for models.py

"""

# Standard packages
import pytest
import numpy as np
from pickle import PickleError
from skopt.space.space import Categorical
from sklearn.ensemble import RandomForestRegressor

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.models import MODELS
from .utils import X_bin, y_bin, X_class2, y_class2, X_reg, y_reg, mnist


# Variables ======================================================== >>

binary, multiclass, regression = [], [], []
for m in MODELS.values():
    if m.task != "reg":
        if m.acronym != "CatNB":
            binary.append(m.acronym)
        if not m.acronym.startswith("Cat"):
            multiclass.append(m.acronym)
    if m.task != "class":
        regression.append(m.acronym)


# Functions ======================================================= >>

def neural_network():
    """Returns a convolutional neural network."""

    def create_model():
        """Returns a convolutional neural network."""
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
        model.add(Conv2D(64, kernel_size=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        return model

    return KerasClassifier(create_model, epochs=1, batch_size=512, verbose=0)


# Test custom models =============================================== >>

@pytest.mark.parametrize("model", [RandomForestRegressor, RandomForestRegressor()])
def test_custom_models(model):
    """Assert that ATOM works with custom models."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=model, n_calls=2, n_initial_points=1)
    assert atom.rfr.fullname == "RandomForestRegressor"
    assert atom.rfr.estimator.get_params()["random_state"] == 1


def test_deep_learning_models():
    """Assert that ATOM works with deep learning models."""
    atom = ATOMClassifier(*mnist, n_rows=0.05, random_state=1)
    pytest.raises(PermissionError, atom.clean)
    atom.run(models=neural_network())
    assert atom.models == "KC"  # KerasClassifier


def test_nice_error_for_unpickable_models():
    """Assert that pickle errors raise an understandable exception."""
    atom = ATOMClassifier(*mnist, n_rows=0.05, n_jobs=2, random_state=1)
    pytest.raises(
        PickleError,
        atom.run,
        models=neural_network(),
        n_calls=5,
        bo_params={
            "cv": 3,
            "dimensions": [Categorical([64, 128, 256], name="batch_size")],
        },
    )


# Test predefined models =========================================== >>

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


def test_CatNB():
    """Assert that the CatNB model works. Separated because of special dataset."""
    X = np.random.randint(5, size=(100, 100))
    y = np.random.randint(2, size=100)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CatNB", n_calls=2, n_initial_points=1)
    assert not atom.errors
    assert hasattr(atom, "CatNB")
