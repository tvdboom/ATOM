"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for the utils module.

"""

from datetime import timedelta
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from atom import show_versions
from atom.pipeline import Pipeline
from atom.utils.patches import VotingClassifier, VotingRegressor
from atom.utils.utils import (
    ClassMap, check_is_fitted, time_to_str, to_df, to_series,
)

from .conftest import X_bin, X_reg, y_bin, y_reg


@pytest.fixture()
def classifiers():
    """Get a list of classifiers for the ensemble."""
    return [
        ("lda", LinearDiscriminantAnalysis().fit(X_bin, y_bin)),
        ("placeholder1", "drop"),
        (
            "pl",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("et", ExtraTreesClassifier(n_estimators=5)),
                ],
            ).fit(X_bin, y_bin),
        ),
    ]


@pytest.fixture()
def regressors():
    """Get a list of regressors for the ensemble."""
    return [
        ("ols", LinearRegression()),
        ("placeholder1", "drop"),
        (
            "pl",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("et", ExtraTreesRegressor(n_estimators=5)),
                ],
            ),
        ),
    ]


# Test _show_versions ============================================== >>

@patch.dict("sys.modules", {"sklearn": "1.3.2"}, clear=True)
def test_show_versions():
    """Assert that the show_versions function runs without errors."""
    show_versions()


# Test patches ===================================================== >>

def test_voting_initialized_fitted(classifiers):
    """Assert that the model can be fit at initialization."""
    vote = VotingClassifier(estimators=classifiers)
    assert check_is_fitted(vote, exception=False)


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
    assert not check_is_fitted(vote, exception=False)
    vote.fit(X_bin, y_bin)
    assert check_is_fitted(vote, exception=False)
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
    assert not check_is_fitted(vote, exception=False)
    vote.fit(X_reg, y_reg)
    assert check_is_fitted(vote, exception=False)


# Test utils ======================================================= >>

def test_classmap_failed_initialization():
    """Assert that an error is raised when the classes do not have the key attribute."""
    with pytest.raises(ValueError, match=".*has no attribute.*"):
        ClassMap(2, 3)


def test_classmap_manipulations():
    """Assert that the ClassMap class can be manipulated."""
    cm = ClassMap(2, 3, 4, 5, 6, key="real")
    assert str(cm[[3, 4]]) == "[3, 4]"
    assert str(cm[3:5]) == "[5, 6]"
    assert str(cm[3]) == "3"
    cm[2] = 8
    assert str(cm) == "[2, 3, 8, 5, 6]"
    del cm[4]
    assert str(cm) == "[2, 3, 8, 5]"
    assert str(list(reversed(cm))) == "[5, 8, 3, 2]"
    cm += [6]
    assert str(cm) == "[2, 3, 8, 5, 6]"
    assert cm.index(3) == 1
    cm.clear()
    assert str(cm) == "[]"


def test_time_to_string():
    """Assert that the time strings are formatted properly."""
    assert time_to_str(timedelta(seconds=17).total_seconds()).startswith("17.00")
    assert time_to_str(timedelta(minutes=1, seconds=2).total_seconds()) == "01m:02s"
    assert time_to_str(timedelta(hours=3, minutes=8).total_seconds()) == "03h:08m:00s"


def test_to_tabular_with_cuml():
    """Assert that cuML objects use the to_tabular method."""
    to_df(Mock(spec=["to_tabular"]), columns=[0, 1])
    to_series(Mock(spec=["to_tabular"]))
