# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for ensembles.py

"""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from atom.ensembles import (
    StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor,
)
from atom.pipeline import Pipeline
from atom.utils.utils import check_is_fitted

from .conftest import X_bin, X_reg, y_bin, y_reg


@pytest.fixture
def classifiers():
    """Get a list of classifiers for the ensemble."""
    return [
        ("lda", LinearDiscriminantAnalysis().fit(X_bin, y_bin)),
        ("placeholder1", "drop"),
        ("pl", Pipeline(
            [("scaler", StandardScaler()), ("et", ExtraTreesClassifier(n_estimators=5))]
        ).fit(X_bin, y_bin)),
    ]


@pytest.fixture
def regressors():
    """Get a list of regressors for the ensemble."""
    return [
        ("ols", LinearRegression()),
        ("placeholder1", "drop"),
        ("pl", Pipeline(
            [("scaler", StandardScaler()), ("et", ExtraTreesRegressor(n_estimators=5))]
        )),
    ]


# Stacking ========================================================= >>

def test_stacking_classifier(classifiers):
    """Assert that stacking classifiers work."""
    stack = StackingClassifier(estimators=classifiers, cv=KFold())
    assert not check_is_fitted(stack, False)
    stack.fit(X_bin, y_bin)
    assert check_is_fitted(stack, False)
    assert len(stack.estimators_) == 2
    assert stack.estimators_[0] is classifiers[0][1]  # Fitted is same
    assert stack.estimators_[1] is not classifiers[1][1]  # Unfitted changes


def test_stacking_regressor(regressors):
    """Assert that stacking regressors."""
    stack = StackingRegressor(estimators=regressors)
    assert not check_is_fitted(stack, False)
    stack.fit(X_reg, y_reg)
    assert check_is_fitted(stack, False)
    assert len(stack.estimators_) == 2


# Voting =========================================================== >>

def test_voting_initialized_fitted(classifiers):
    """Assert that the model can be fit at initialization."""
    vote = VotingClassifier(estimators=classifiers)
    assert check_is_fitted(vote, False)


def test_voting_multilabel(classifiers):
    """Assert that an error is raised for multilabel targets."""
    vote = VotingClassifier(estimators=classifiers)
    with pytest.raises(NotImplementedError, match=".*Multilabel.*"):
        vote.fit(X_bin, np.array([[0, 1], [1, 0]]))


def test_voting_invalid_type(classifiers):
    """Assert that an error is raised when voting type is invalid."""
    vote = VotingClassifier(estimators=classifiers, voting="invalid")
    with pytest.raises(ValueError, match=".*must be 'soft'.*"):
        vote.fit(X_bin, y_bin)


def test_voting_invalid_weights(classifiers):
    """Assert that an error is raised when weights have invalid length."""
    vote = VotingClassifier(estimators=classifiers, weights=[0, 1])
    with pytest.raises(ValueError, match=".*estimators and weights.*"):
        vote.fit(X_bin, y_bin)


def test_voting_mixed_fit_and_not(classifiers):
    """Assert that fitted and non-fitted models can be used both."""
    estimators = classifiers.copy()
    estimators.append(("not_fitted_lda", LinearDiscriminantAnalysis()))

    vote = VotingClassifier(estimators=estimators)
    assert not check_is_fitted(vote, False)
    vote.fit(X_bin, y_bin)
    assert check_is_fitted(vote, False)
    assert len(vote.estimators_) == 3
    assert vote.estimators_[0] is estimators[0][1]  # Fitted is same
    assert vote.estimators_[2] is not estimators[2][1]  # Unfitted changes


@pytest.mark.parametrize("voting", ["soft", "hard"])
def test_voting_predict(classifiers, voting):
    """Assert that the predict method doesn't use the encoder."""
    vote = VotingClassifier(estimators=classifiers, voting=voting)
    assert isinstance(vote.predict(X_bin), np.ndarray)


def test_voting_regressor(regressors):
    """Assert that the regressor works."""
    vote = VotingRegressor(estimators=regressors)
    assert not check_is_fitted(vote, False)
    vote.fit(X_reg, y_reg)
    assert check_is_fitted(vote, False)
