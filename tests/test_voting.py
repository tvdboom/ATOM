# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for voting.py

"""

# Standard packages
import pytest
import numpy as np

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.training import DirectClassifier
from atom.utils import NotFittedError
from .utils import X_bin, y_bin, X_reg, y_reg


def test_vote_as_attribute():
    """Assert that the vote model is attached as attribute."""
    trainer = DirectClassifier("LGB")
    assert hasattr(trainer, "vote")


def test_vote_repr():
    """Assert that the vote model has a __repr__."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert str(atom.vote).startswith("VotingClassifier")


def test_vote_predict():
    """Assert that the predict method returns the majority voting."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.vote.predict, X_reg)
    atom.run(models=["XGB", "LGB"])
    assert isinstance(atom.vote.predict(X_reg), np.ndarray)


def test_vote_predict_proba():
    """Assert that the predict_proba method returns the average predictions."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.vote.predict_proba, X_bin)
    atom.run(models=["XGB", "LGB"])
    assert isinstance(atom.vote.predict_proba(X_bin), np.ndarray)


def test_vote_scoring():
    """Assert that the scoring method returns the average predictions."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models=["RF", "XGB", "LGB"], metric=["f1", "recall"])
    assert atom.vote.scoring() == "f1: 0.978   recall: 1.000"
