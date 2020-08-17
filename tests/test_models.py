# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for models.py

"""

# Import packages
import pytest
from atom import ATOMClassifier, ATOMRegressor
from atom.models import MODEL_LIST
from atom.utils import ONLY_CLASS, ONLY_REG
from .utils import X_bin, y_bin, X_class2, y_class2, X_reg, y_reg


# Tests ===================================================================== >>

@pytest.mark.parametrize('model', [m for m in MODEL_LIST if m not in ONLY_REG])
def test_models_binary(model):
    """Assert that the fit method works with all models for binary."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.24, random_state=1)
    atom.run(models=model,
             metric='auc',
             n_calls=2,
             n_random_starts=1,
             bo_params={'base_estimator': 'rf', 'cv': 1})
    assert not atom.errors  # Assert that the model ran without errors
    assert hasattr(atom, model) and hasattr(atom, model.lower())


@pytest.mark.parametrize('model', [m for m in MODEL_LIST if m not in ONLY_REG])
def test_models_multiclass(model):
    """Assert that the fit method works with all models for multiclass."""
    if model != 'CatB':  # CatBoost fails with an unclear error
        atom = ATOMClassifier(X_class2, y_class2, test_size=0.24, random_state=1)
        atom.run(models=model,
                 metric='f1_micro',
                 n_calls=2,
                 n_random_starts=1,
                 bo_params={'base_estimator': 'rf', 'cv': 1})
        assert not atom.errors
        assert hasattr(atom, model) and hasattr(atom, model.lower())


@pytest.mark.parametrize('model', [m for m in MODEL_LIST if m not in ONLY_CLASS])
def test_models_regression(model):
    """Assert that the fit method works with all models for regression."""
    atom = ATOMRegressor(X_reg, y_reg, test_size=0.24, random_state=1)
    atom.run(models=model,
             metric='neg_mean_absolute_error',
             n_calls=2,
             n_random_starts=1,
             bo_params={'base_estimator': 'gbrt', 'cv': 1})
    assert not atom.errors
    assert hasattr(atom, model) and hasattr(atom, model.lower())
