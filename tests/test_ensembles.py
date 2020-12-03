# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for ensembles.py

"""

# Standard packages
import pytest
import numpy as np

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.utils import NotFittedError
from .utils import X_bin, y_bin, X_reg, y_reg


# Voting =========================================================== >>

def test_vote_name():
    """Assert that the vote gets the name depending on the goal."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.vote.fullname == "VotingClassifier"

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    assert atom.vote.fullname == "VotingRegressor"


def test_vote_repr():
    """Assert that the vote model has a __repr__."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert str(atom.vote).startswith("VotingClassifier")


def test_vote_weights_getter():
    """Assert that the weights property can be get."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.vote.weights is None


def test_vote_weights_setter_invalid():
    """Assert that an error is raised for invalid weight length."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"])
    with pytest.raises(ValueError, match=r".*weights should have.*"):
        atom.vote.weights = [2, 3, 4]


def test_vote_weights_setter():
    """Assert that the weights property can be set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"])
    atom.vote.weights = [2, 3]
    assert atom.vote.weights == [2, 3]


def test_vote_exclude_getter():
    """Assert that the weights property can be get."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"])
    assert atom.vote.exclude == []


def test_vote_exclude_setter():
    """Assert that the weights property can be set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"])
    atom.vote.exclude = ["Tree"]
    assert atom.vote.models == ["LGB"]


def test_vote_scoring():
    """Assert that the scoring method returns the average score."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"], metric=["precision", "recall"])
    assert atom.vote.scoring() == "precision: 0.944   recall: 0.993"

    avg = (atom.tree.scoring("f1") + atom.lgb.scoring("f1"))/2
    assert atom.vote.scoring("f1") == avg


def test_vote_scoring_with_weights():
    """Assert that the scoring works with weights."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"])
    atom.vote.weights = [1, 2]

    avg = (atom.tree.scoring("f1") + 2 * atom.lgb.scoring("f1"))/3
    assert atom.vote.scoring("f1") == avg


def test_vote_predict():
    """Assert that the predict method works as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.vote.predict, X_reg)
    atom.run(models=["Tree", "LGB"])
    assert isinstance(atom.vote.predict(X_reg), np.ndarray)


def test_vote_predict_proba():
    """Assert that the predict_proba method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.vote.predict_proba, X_bin)
    atom.run(models=["Tree", "LGB"])
    assert isinstance(atom.vote.predict_proba(X_bin), np.ndarray)


def test_vote_predict_log_proba():
    """Assert that the predict_log_proba method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.vote.predict_log_proba, X_bin)
    atom.run(models=["Tree", "RF"])
    assert isinstance(atom.vote.predict_log_proba(X_bin), np.ndarray)


def test_vote_score():
    """Assert that the score method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.vote.score, X_bin, y_bin)
    atom.run(models=["Tree", "LGB"])
    assert isinstance(atom.vote.score(X_bin, y_bin), np.float64)


def test_vote_invalid_method():
    """Assert that an error is raised when a model doesn't have the method."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models=["Tree", "LGB"])
    pytest.raises(AttributeError, atom.vote.predict_log_proba, X_bin)


def test_vote_branch_transformation():
    """Assert that the branches transform every estimator only once."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    atom.branch = "branch_2"
    atom.balance()
    atom.run(models=["Tree", "LGB"])
    atom.vote.predict(X_bin)
    assert not atom.errors


def test_vote_reset_predictions():
    """Assert that the reset_predictions method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models=["Tree", "LGB"])
    print(atom.vote.predict_test)
    atom.vote.reset_predictions()
    assert atom.vote._pred_attrs[1] is None


def test_vote_prediction_attrs():
    """Assert that the prediction attributes can be calculated."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models=["Tree", "RF"])
    assert isinstance(atom.vote.predict_train, np.ndarray)
    assert isinstance(atom.vote.predict_test, np.ndarray)
    assert isinstance(atom.vote.predict_proba_train, np.ndarray)
    assert isinstance(atom.vote.predict_proba_test, np.ndarray)
    assert isinstance(atom.vote.predict_log_proba_train, np.ndarray)
    assert isinstance(atom.vote.predict_log_proba_test, np.ndarray)
    assert isinstance(atom.vote.score_train, np.float64)
    assert isinstance(atom.vote.score_test, np.float64)
