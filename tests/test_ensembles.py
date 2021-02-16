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
from atom.utils import check_scaling
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg


# Voting =========================================================== >>

def test_vote_one_model():
    """Assert that an error is raised for voting with one model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR"])
    pytest.raises(ValueError, atom.voting)


def test_vote_unequal_weight_length():
    """Assert that an error is raised if the weights have not len(models)."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    pytest.raises(ValueError, atom.voting, weights=[1, 2, 3])


def test_vote_repr():
    """Assert that the vote model has a __repr__."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree"])
    atom.voting()
    assert str(atom.vote).startswith("Voting")


def test_vote_scoring():
    """Assert that the scoring method returns the average score."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"], metric=["precision", "recall"])
    atom.voting()
    assert atom.vote.scoring() == "precision: 0.944   recall: 0.993"


def test_vote_scoring_with_weights():
    """Assert that the scoring works with weights."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"])
    atom.voting(weights=[1, 2])
    avg = (atom.tree.scoring("r2") + 2 * atom.lgb.scoring("r2"))/3
    assert atom.vote.scoring("r2") == avg


def test_vote_invalid_method():
    """Assert that an error is raised when a model doesn't have the method."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models=["Tree", "LGB"])
    atom.voting()
    pytest.raises(AttributeError, atom.vote.predict_log_proba, X_bin)


def test_vote_branch_transformation():
    """Assert that the branches transform every estimator only once."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.impute()
    atom.branch = "branch_2"
    atom.balance()
    atom.run(models=["Tree", "LGB"])
    atom.voting()
    assert isinstance(atom.vote.predict(X_bin), np.ndarray)


def test_vote_prediction_methods():
    """Assert that the prediction methods work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.run(models=["Tree"])
    atom.branch = "branch_2"
    atom.impute(strat_num="mean", strat_cat="most_frequent")
    atom.run(["LGB"])
    atom.voting(models=["Tree", "LGB"])
    pytest.raises(AttributeError, atom.vote.decision_function, X_bin)
    assert isinstance(atom.vote.predict(X_bin), np.ndarray)
    assert isinstance(atom.vote.predict_proba(X_bin), np.ndarray)
    assert isinstance(atom.vote.score(X_bin, y_bin), np.float64)


def test_vote_prediction_attrs():
    """Assert that the prediction attributes can be calculated."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models=["Tree", "RF"])
    atom.voting()
    assert isinstance(atom.vote.metric_train, np.float64)
    assert isinstance(atom.vote.metric_test, np.float64)
    assert isinstance(atom.vote.predict_train, np.ndarray)
    assert isinstance(atom.vote.predict_test, np.ndarray)
    assert isinstance(atom.vote.predict_proba_train, np.ndarray)
    assert isinstance(atom.vote.predict_proba_test, np.ndarray)
    assert isinstance(atom.vote.predict_log_proba_train, np.ndarray)
    assert isinstance(atom.vote.predict_log_proba_test, np.ndarray)
    assert isinstance(atom.vote.score_train, np.float64)
    assert isinstance(atom.vote.score_test, np.float64)


# Stacking ========================================================= >>


def test_stack_one_model():
    """Assert that an error is raised for stacking with one model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR"])
    pytest.raises(ValueError, atom.stacking)


def test_stack_method():
    """Assert that we can customize the stack method."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["Tree", "PA"])
    atom.stacking(stack_method="predict")
    assert atom.stack.shape == (len(X_class), 3)


def test_passthrough():
    """Assert that the features are added when passthrough=True."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["Tree", "LGB"])
    atom.stacking(passthrough=True)
    assert atom.stack.shape == (atom.shape[0], 6 + atom.shape[1])


def test_passthrough_scaled():
    """Assert that the features are scaled when models need it."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["Tree", "LGB"])
    atom.stacking(passthrough=True)
    assert check_scaling(atom.stack.X.iloc[:, 2:])


def test_passthrough_not_scaled():
    """Assert that the features are not scaled when models don't need it."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["Tree", "RF"])
    atom.stacking(passthrough=True)
    assert not check_scaling(atom.stack.X.iloc[:, 2:])


def test_predefined_model():
    """Assert that you can use a predefined model as final estimator."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "PA"])
    atom.stacking(estimator="RF")
    assert atom.stack.estimator.__class__.__name__ == "RandomForestClassifier"


def test_stack_repr():
    """Assert that the stack model has a __repr__."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree"])
    atom.stacking()
    assert str(atom.stack).startswith("Stacking")


def test_stack_scoring():
    """Assert that the scoring method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, verbose=2, random_state=1)
    atom.run(["Tree", "RF"])
    atom.stacking()
    assert atom.stack.scoring() == "f1: 0.957"
    assert atom.stack.scoring("recall") == 0.9852941176470589


def test_stack_predictions_binary():
    """Assert that the prediction methods work for binary tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "PA"])
    atom.stacking(models=["Tree", "PA"], passthrough=True)
    pytest.raises(AttributeError, atom.stack.decision_function, X_bin)
    assert isinstance(atom.stack.predict(X_bin), np.ndarray)
    assert isinstance(atom.stack.score(X_bin, y_bin), np.float64)


def test_stack_predictions_multiclass():
    """Assert that the prediction methods work for multiclass tasks."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["Tree", "PA"])
    atom.stacking(models=["Tree", "PA"], passthrough=True)
    assert isinstance(atom.stack.predict(X_class), np.ndarray)
    assert isinstance(atom.stack.score(X_class, y_class), np.float64)


def test_stack_predictions_regression():
    """Assert that the prediction methods work for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.clean()
    atom.run(models=["Tree"])
    atom.branch = "branch_2"
    atom.impute(strat_num="mean", strat_cat="most_frequent")
    atom.run(["PA"])
    atom.stacking(models=["Tree", "PA"], passthrough=True)
    assert isinstance(atom.stack.predict(X_reg), np.ndarray)
    assert isinstance(atom.stack.score(X_reg, y_reg), np.float64)
