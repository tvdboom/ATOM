# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for models.py

"""

# Import packages
from atom import ATOMClassifier, ATOMRegressor
from atom.models import MODEL_LIST
from atom.utils import ONLY_CLASSIFICATION, ONLY_REGRESSION
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg


# << ================= Tests ================ >>

def test_models_attributes():
    """Assert that model subclasses have the right attributes."""
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(models='lr', metric='f1')
    assert atom.lr.name == 'LR'
    assert atom.lr.longname == 'Logistic Regression'


def test_models_binary():
    """Assert that the fit method works with all models for binary."""
    for model in [m for m in MODEL_LIST.keys() if m not in ONLY_REGRESSION]:
        atom = ATOMClassifier(X_bin, y_bin, test_size=0.24, random_state=1)
        atom.pipeline(models=model,
                      metric='f1',
                      n_calls=2,
                      n_random_points=1,
                      bo_kwargs={'cv': 1})
        assert not atom.errors  # Assert that the model ran without errors


def test_models_multiclass():
    """Assert that the fit method works with all models for multiclass."""
    for model in [m for m in MODEL_LIST.keys() if m not in ONLY_REGRESSION]:
        atom = ATOMClassifier(X_class, y_class, test_size=0.24, random_state=1)
        atom.pipeline(models=model,
                      metric='f1_micro',
                      n_calls=2,
                      n_random_points=1,
                      bo_kwargs={'cv': 1})
        assert not atom.errors


def test_models_regression():
    """Assert that the fit method works with all models for regression."""
    for model in [m for m in MODEL_LIST.keys() if m not in ONLY_CLASSIFICATION]:
        atom = ATOMRegressor(X_reg, y_reg, test_size=0.24, random_state=1)
        atom.pipeline(models=model,
                      metric='neg_mean_absolute_error',
                      n_calls=2,
                      n_random_points=1,
                      bo_kwargs={'cv': 1})
        assert not atom.errors
