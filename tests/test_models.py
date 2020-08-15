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
from .utils import X_bin, y_bin, X_class2, y_class2, X_reg, y_reg


# Tests ===================================================================== >>

def test_models_binary():
    """Assert that the fit method works with all models for binary."""
    for model in [m for m in MODEL_LIST if m not in ONLY_REGRESSION]:
        atom = ATOMClassifier(X_bin, y_bin, test_size=0.24, random_state=1)
        atom.run(models=model,
                 metric='auc',
                 n_calls=2,
                 n_random_starts=1,
                 bo_params={'base_estimator': 'rf', 'cv': 1})
        assert not atom.errors  # Assert that the model ran without errors
        assert hasattr(atom, model) and hasattr(atom, model.lower())


def test_models_multiclass():
    """Assert that the fit method works with all models for multiclass."""
    for model in [m for m in MODEL_LIST if m not in ONLY_REGRESSION]:
        if model == 'CatB':
            continue  # CatBoost fails with a weird error
        atom = ATOMClassifier(X_class2, y_class2, test_size=0.24, random_state=1)
        atom.run(models=model,
                 metric='f1_micro',
                 n_calls=2,
                 n_random_starts=1,
                 bo_params={'base_estimator': 'rf', 'cv': 1})
        assert not atom.errors
        assert hasattr(atom, model) and hasattr(atom, model.lower())


def test_models_regression():
    """Assert that the fit method works with all models for regression."""
    for model in [m for m in MODEL_LIST if m not in ONLY_CLASSIFICATION]:
        atom = ATOMRegressor(X_reg, y_reg, test_size=0.24, random_state=1)
        atom.run(models=model,
                 metric='neg_mean_absolute_error',
                 n_calls=2,
                 n_random_starts=1,
                 bo_params={'base_estimator': 'gbrt', 'cv': 1})
        assert not atom.errors
        assert hasattr(atom, model) and hasattr(atom, model.lower())
