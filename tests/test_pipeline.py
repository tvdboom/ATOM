# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for pipeline.py

"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from atom import ATOMClassifier
from atom.pipeline import Pipeline

from .conftest import X_bin, y_bin


@pytest.fixture
def pipeline():
    """Get a pipeline from atom with/without final estimator."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    atom.prune()
    atom.add(StandardScaler(), columns=0)
    atom.run("LR")

    def get_pipeline(model, memory=None):
        return atom.export_pipeline(model="LR" if model else None, memory=memory)

    return get_pipeline


def test_fit(pipeline):
    """Assert that the pipeline can be fitted normally."""
    pl = pipeline(model=True)
    assert pl.fit(X_bin, y_bin)
    pl.steps.insert(1, ("passthrough", None))
    assert pl.fit(X_bin, y_bin)


def test_internal_attrs_are_saved(pipeline):
    """Assert that cols and train_only attrs are stored after clone."""
    pl = pipeline(model=False, memory=True)
    pl.fit(X_bin, y_bin)
    assert pl.steps[-1][1]._cols == [["mean radius"], []]
    assert pl.steps[-2][1]._train_only is True


def test_transform_only_X_or_y(pipeline):
    """Assert that the pipeline can transform only X or y."""
    pl = Pipeline(
        steps=[
            ("label_encoder", LabelEncoder()),
            ("scaler", StandardScaler()),
        ]
    )
    pl.fit(X_bin, y_bin)
    assert isinstance(pl.transform(y=y_bin), pd.Series)
    assert isinstance(pl.transform(X=X_bin), pd.DataFrame)


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
