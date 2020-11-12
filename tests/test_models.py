# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for models.py

"""

# Standard packages
import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.wrappers.scikit_learn import KerasClassifier

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.models import MODEL_LIST
from atom.utils import ONLY_CLASS, ONLY_REG
from .utils import X_bin, y_bin, X_class2, y_class2, X_reg, y_reg


# Variables ======================================================== >>

binary = [m for m in MODEL_LIST if m not in ["custom", "CatNB"] + ONLY_REG]
multiclass = [m for m in MODEL_LIST if m not in ["custom", "CatNB", "CatB"] + ONLY_REG]
regression = [m for m in MODEL_LIST if m not in ["custom"] + ONLY_CLASS]


# Functions ======================================================= >>

def neural_network():
    """Create a convolutional neural network in Keras."""
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model

# Tests =========================================================== >>


@pytest.mark.parametrize("model", [RandomForestRegressor, RandomForestRegressor()])
def test_custom_models(model):
    """Assert that ATOM works with custom models."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=model, n_calls=2, n_initial_points=1)
    assert not atom.errors
    assert hasattr(atom, "RandomForestRegressor")
    assert atom.RandomForestRegressor.estimator.get_params()["random_state"] == 1


def test_deep_learning_models():
    """Assert that ATOM works with deep learning models."""
    # Since ATOM uses sklearn's API, use Keras' wrapper
    model = KerasClassifier(neural_network, epochs=1, batch_size=512, verbose=0)

    # Download the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape data to fit model
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    atom = ATOMClassifier((X_train, y_train), (X_test, y_test), random_state=1)
    pytest.raises(PermissionError, atom.clean)
    atom.run(models=model)
    assert not atom.errors


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
    assert not atom.errors  # Assert that the model ran without errors
    assert hasattr(atom, model) and hasattr(atom, model.lower())


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
    assert hasattr(atom, model) and hasattr(atom, model.lower())


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
    assert hasattr(atom, model) and hasattr(atom, model.lower())


def test_CatNB():
    """Assert that the CatNB model works. Separated because of special dataset."""
    X = np.random.randint(5, size=(100, 100))
    y = np.random.randint(2, size=100)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CatNB", n_calls=2, n_initial_points=1)
    assert not atom.errors
    assert hasattr(atom, "CatNB") and hasattr(atom, "catnb")
