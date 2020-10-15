# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for models.py

"""

# Import packages
import pytest
import numpy as np
from atom import ATOMClassifier, ATOMRegressor
from atom.models import MODEL_LIST
from atom.utils import ONLY_CLASS, ONLY_REG
from .utils import X_bin, y_bin, X_class2, y_class2, X_reg, y_reg


# Variables ================================================================= >>

binary = [m for m in MODEL_LIST if m not in ONLY_REG and m != 'CatNB']
multiclass = [m for m in MODEL_LIST if m not in ONLY_REG and m not in ['CatNB', 'CatB']]
regression = [m for m in MODEL_LIST if m not in ONLY_CLASS]


# Tests ===================================================================== >>

@pytest.mark.parametrize('model', binary)
def test_models_binary(model):
    """Assert that all models work with binary classification."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.24, random_state=1)
    atom.run(models=model,
             metric='auc',
             n_calls=2,
             n_initial_points=1,
             bo_params={'base_estimator': 'rf', 'cv': 1})
    assert not atom.errors  # Assert that the model ran without errors
    assert hasattr(atom, model) and hasattr(atom, model.lower())


@pytest.mark.parametrize('model', multiclass)
def test_models_multiclass(model):
    """Assert that all models work with multiclass classification."""
    atom = ATOMClassifier(X_class2, y_class2, test_size=0.24, random_state=1)
    atom.run(models=model,
             metric='f1_micro',
             n_calls=2,
             n_initial_points=1,
             bo_params={'base_estimator': 'rf', 'cv': 1})
    assert not atom.errors
    assert hasattr(atom, model) and hasattr(atom, model.lower())


@pytest.mark.parametrize('model', regression)
def test_models_regression(model):
    """Assert that all models work with regression."""
    atom = ATOMRegressor(X_reg, y_reg, test_size=0.24, random_state=1)
    atom.run(models=model,
             metric='neg_mean_absolute_error',
             n_calls=2,
             n_initial_points=1,
             bo_params={'base_estimator': 'gbrt', 'cv': 1})
    assert not atom.errors
    assert hasattr(atom, model) and hasattr(atom, model.lower())


def test_CatNB():
    """Assert that the CatNB model works. Separated because of special dataset."""
    X = np.random.randint(5, size=(100, 100))
    y = np.random.randint(2, size=100)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models='CatNB', n_calls=2, n_initial_points=1)
    assert not atom.errors
    assert hasattr(atom, 'CatNB') and hasattr(atom, 'catnb')
