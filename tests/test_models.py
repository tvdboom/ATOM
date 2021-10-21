# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for models.py

"""

# Standard packages
import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.models import MODEL_LIST
from .utils import X_bin, y_bin, X_class2, y_class2, X_reg, y_reg, mnist


# Variables ======================================================== >>

binary = [m for m in MODEL_LIST if m.task != "reg" and m.acronym != "CatNB"]
multi = [m for m in MODEL_LIST if m.task != "reg" and not m.acronym.startswith("Cat")]
regression = [m for m in MODEL_LIST if m.task != "class"]


# Functions ======================================================= >>

def neural_network():
    """Create a convolutional neural network in Keras."""
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model


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
    atom = ATOMClassifier(*mnist, n_rows=0.1, random_state=1)
    pytest.raises(PermissionError, atom.clean)
    atom.run(KerasClassifier(neural_network, epochs=1, batch_size=512, verbose=0))


# Test predefined models =========================================== >>

@pytest.mark.parametrize("model", binary)
def test_models_binary(model):
    """Assert that all models work with binary classification."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.24, random_state=1)
    atom.run(
        models=model.acronym,
        metric="auc",
        n_calls=2,
        n_initial_points=1,
        bo_params={"base_estimator": "rf", "cv": 1},
    )
    assert not atom.errors  # The model ran without errors
    assert hasattr(atom, model.acronym)  # The model is an attr of the trainer


@pytest.mark.parametrize("model", multi)
def test_models_multiclass(model):
    """Assert that all models work with multiclass classification."""
    atom = ATOMClassifier(X_class2, y_class2, test_size=0.24, random_state=1)
    atom.run(
        models=model.acronym,
        metric="f1_micro",
        n_calls=2,
        n_initial_points=1,
        bo_params={"base_estimator": "rf", "cv": 1},
    )
    assert not atom.errors
    assert hasattr(atom, model.acronym)


@pytest.mark.parametrize("model", regression)
def test_models_regression(model):
    """Assert that all models work with regression."""
    atom = ATOMRegressor(X_reg, y_reg, test_size=0.24, random_state=1)
    atom.run(
        models=model.acronym,
        metric="neg_mean_absolute_error",
        n_calls=2,
        n_initial_points=1,
        bo_params={"base_estimator": "gbrt", "cv": 1},
    )
    assert not atom.errors
    assert hasattr(atom, model.acronym)


def test_CatNB():
    """Assert that the CatNB model works. Separated because of special dataset."""
    X = np.random.randint(5, size=(100, 100))
    y = np.random.randint(2, size=100)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CatNB", n_calls=2, n_initial_points=1)
    assert not atom.errors
    assert hasattr(atom, "CatNB")
