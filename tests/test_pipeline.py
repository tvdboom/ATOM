# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for pipeline.py

"""

# Standard packages
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Own modules
from atom import ATOMClassifier
from atom.pipeline import Pipeline
from .utils import X_bin, y_bin


@pytest.fixture
def pipeline():
    """Get a pipeline from atom with/without final estimator."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    atom.prune()
    atom.add(StandardScaler())
    atom.run("LR")

    def get_pipeline(model):
        return atom.export_pipeline(model="LR" if model else None)

    return get_pipeline


def test_fit(pipeline):
    """Assert that the pipeline can be fitted normally."""
    pl = pipeline(model=True)
    assert pl.fit(X_bin, y_bin)
    pl.steps.insert(1, ("passthrough", None))
    assert pl.fit(X_bin, y_bin)


def test_transform_only_y(pipeline):
    """Assert that the pipeline can transform the target column only."""
    pl = Pipeline(steps=[("label_encoder", LabelEncoder())])
    assert isinstance(pl.fit_transform(y=y_bin), pd.Series)


def test_fit_transform(pipeline):
    """Assert that the pipeline can be fit-transformed normally."""
    pl = pipeline(model=False)
    pl.steps[0] = ("test", "passthrough")
    assert isinstance(pl.fit_transform(X_bin), pd.DataFrame)  # Returns X
    pl.steps[-1] = ("test_final", "passthrough")
    assert isinstance(pl.fit_transform(X_bin, y_bin), tuple)  # Returns X, y


def test_transform_train_only(pipeline):
    """Assert that the pipeline ignores train_only during predicting."""
    pl = pipeline(model=False)
    assert len(pl.transform(X_bin)) == len(X_bin)  # Pruner is not applied


def test_predict(pipeline):
    """Assert that the pipeline uses predict normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.predict(X_bin), np.ndarray)


def test_predict_proba(pipeline):
    """Assert that the pipeline uses predict_proba normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.predict_proba(X_bin), np.ndarray)


def test_predict_log_proba(pipeline):
    """Assert that the pipeline uses predict_log_proba normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.predict_log_proba(X_bin), np.ndarray)


def test_decision_function(pipeline):
    """Assert that the pipeline uses decision_function normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.decision_function(X_bin), np.ndarray)


def test_score(pipeline):
    """Assert that the pipeline uses score normally."""
    pl = pipeline(model=True)
    assert isinstance(pl.score(X_bin, y_bin), np.float64)


def test_transform(pipeline):
    """Assert that the pipeline uses transform normally."""
    pl = pipeline(model=False)
    assert isinstance(pl.transform(X_bin), pd.DataFrame)
    assert isinstance(pl.transform(X_bin, y_bin), tuple)
