# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for basepredictor.py

"""

# Standard packages
import pytest
import numpy as np

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.branch import Branch
from atom.training import DirectClassifier
from atom.utils import NotFittedError
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg, bin_train


# Test magic methods =============================================== >>

def test_getattr_from_branch():
    """Assert that branch attributes can be called from the trainer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.pipeline is atom.branch.pipeline


def test_getattr_model():
    """Assert that the models can be called as attributes from the trainer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom.tree is atom._models[0]


def test_getattr_invalid():
    """Assert that an error is raised when there is no such attribute."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(AttributeError, match=r".*object has no attribute.*"):
        _ = atom.invalid


def test_setattr_to_branch():
    """Assert that branch properties can be set from the trainer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.dataset = bin_train
    assert atom.shape == (398, 31)


def test_setattr_normal():
    """Assert that trainer attributes can be set normally."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.attr = "test"
    assert atom.attr == "test"


def test_delattr_models():
    """Assert that models can be deleted through del."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR"])
    del atom.lr
    assert not atom.models


def test_delattr_branch():
    """Assert that branches can be deleted through del."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "branch_2"
    del atom.branch
    assert list(atom._branches.keys()) == ["master"]


def test_delattr_normal():
    """Assert that trainer attributes can be deleted normally."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    del atom._models
    assert not hasattr(atom, "_models")


# Test utility properties ========================================== >>

def test_branch_property():
    """Assert that the branch property returns the current branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom.branch, Branch)


def test_metric_property():
    """Assert that the metric property returns the metric names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="f1")
    assert atom.metric == "f1"


def test_metric_property_no_run():
    """Assert that the metric property doesn't crash for unfitted trainers."""
    trainer = DirectClassifier("lr", metric="r2", random_state=1)
    assert trainer.metric == "r2"


def test_models_property():
    """Assert that the models property returns the model names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree"])
    assert atom.models == ["LR", "Tree"]


def test_models_property_no_run():
    """Assert that the models property doesn't crash for unfitted trainers."""
    trainer = DirectClassifier(["LR", "Tree"], metric="r2", random_state=1)
    assert trainer.models == ["LR", "Tree"]


def test_results_property():
    """Assert that the results property returns an overview of the results."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom.results.shape == (1, 4)


def test_results_property_dropna():
    """Assert that the results property doesn't return columns with NaNs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert "mean_bagging" not in atom.results


def test_results_property_successive_halving():
    """Assert that the results works for successive halving runs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(["LR", "Tree"])
    assert atom.results.shape == (3, 4)
    assert list(atom.results.index.get_level_values(0)) == [0.5, 0.5, 1.0]


def test_results_property_train_sizing():
    """Assert that the results works for train sizing runs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing("LR")
    assert atom.results.shape == (5, 4)
    assert list(atom.results.index.get_level_values(0)) == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_winner_property():
    """Assert that the winner property returns the best model in the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree", "LGB"], n_calls=0)
    assert atom.winner is atom.lgb


# Test prediction methods ========================================== >>

def test_reset_predictions():
    """Assert that we can reset all predictions."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    print(atom.lr.predict_proba_train)
    print(atom.lgb.predict_test)
    atom.reset_predictions()
    assert atom.lr._pred_attrs == [None] * 10


def test_predict_method():
    """Assert that the predict method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.predict, X_bin)  # When not yet fitted
    atom.run("LR")
    assert isinstance(atom.predict(X_bin), np.ndarray)


def test_predict_proba_method():
    """Assert that the predict_proba method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.predict_proba, X_bin)
    atom.run("LR")
    assert isinstance(atom.predict_proba(X_bin), np.ndarray)


def test_predict_log_proba_method():
    """Assert that the predict_log_proba method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.predict_log_proba, X_bin)
    atom.run("LR")
    assert isinstance(atom.predict_log_proba(X_bin), np.ndarray)


def test_decision_function_method():
    """Assert that the decision_function method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.decision_function, X_bin)
    atom.run("LR")
    assert isinstance(atom.decision_function(X_bin), np.ndarray)


def test_score_method():
    """Assert that the score method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.score, X_bin, y_bin)
    atom.run("LR")
    assert isinstance(atom.score(X_bin, y_bin), float)


def test_score_method_sample_weights():
    """Assert that the score method works with sample weights."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    score = atom.score(X_bin, y_bin, sample_weight=list(range(len(y_bin))))
    assert isinstance(score, float)


# Test utility methods ============================================= >>

def test_get_model_name_winner():
    """Assert that the winner is returned when used as name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    assert atom._get_model_name("winner") == ["LGB"]


def test_get_model_name_exact_name():
    """Assert that a single model is returned if the name matches exactly."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LR2"])
    assert atom._get_model_name("lr") == ["LR"]


def test_get_model_name_multiple_models():
    """Assert that a list of models is returned when starting the same."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    assert atom._get_model_name("lr") == ["LR1", "LR2"]


def test_get_model_name_digits():
    """Assert that a list of models is returned if using digits."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(["LR", "ET", "RF", "LGB"])
    assert atom._get_model_name("2") == ["LR2", "ET2"]


def test_get_model_name_invalid():
    """Assert that an error is raised when the model name is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    pytest.raises(ValueError, atom._get_model_name, "invalid")


def test_get_models_empty():
    """Assert that all models are returned when the parameter is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    assert atom._get_models(None) == ["LR1", "LR2"]


def test_get_models_str():
    """Assert that the right model is returned when the parameter is a string."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    assert atom._get_models("Lr1") == ["LR1"]


def test_get_models_list():
    """Assert that the right models are returned when the parameter is a list."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2", "LR3"])
    assert atom._get_models(["LR1", "LR2"]) == ["LR1", "LR2"]


def test_get_models_remove_duplicates():
    """Assert that duplicate models are returned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    assert atom._get_models(["LR1", "LR1"]) == ["LR1"]


def test_voting():
    """Assert that the voting method creates a Vote model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.voting)
    atom.run(["LR", "LGB"])
    atom.voting()
    assert hasattr(atom, "Vote") and hasattr(atom, "vote")
    assert "Vote" in atom.models


def test_voting_models_from_branch():
    """Assert that only the models from the current branch are passed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    atom.branch = "branch_2"
    atom.balance()
    atom.run(["RF", "ET"])
    atom.voting()
    assert atom.vote.models == ["RF", "ET"]


def test_stacking():
    """Assert that the stacking method creates a Stack model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.stacking)
    atom.run(["LR", "LGB"])
    atom.stacking()
    assert hasattr(atom, "Stack") and hasattr(atom, "stack")
    assert "Stack" in atom.models


def test_stacking_models_from_branch():
    """Assert that only the models from the current branch are passed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    atom.branch = "branch_2"
    atom.balance()
    atom.run(["RF", "ET"])
    atom.stacking()
    assert atom.stack.models == ["RF", "ET"]


def test_stacking_default_estimator():
    """Assert that a default estimator is provided per goal."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    atom.stacking()
    assert atom.stack.estimator.__class__.__name__ == "LogisticRegression"

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["Tree", "LGB"])
    atom.stacking()
    assert atom.stack.estimator.__class__.__name__ == "Ridge"


def test_class_weights_invalid_dataset():
    """Assert that an error is raised if invalid value for dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.get_class_weight, "invalid")


@pytest.mark.parametrize("dataset", ["train", "test", "dataset"])
def test_get_class_weights(dataset):
    """Assert that the get_class_weight method returns a dict of the classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    class_weight = atom.get_class_weight(dataset)
    assert list(class_weight.keys()) == [0, 1, 2]


def test_calibrate():
    """Assert that the calibrate method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.calibrate)  # When not yet fitted
    atom.run("LR")
    atom.calibrate()
    assert atom.winner.estimator.__class__.__name__ == "CalibratedClassifierCV"


def test_scoring_metric_is_none():
    """Assert that the scoring method works when metric is None."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.scoring)
    atom.run(["Tree", "RF"])
    atom.run("LGB", bagging=5)  # Test with and without bagging
    atom.scoring()


def test_scoring_metric_is_given():
    """Assert that the scoring method works for a specified metric_."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["GNB", "PA"])
    atom.scoring("logloss")  # For _ProbaScorer
    atom.scoring("ap")  # For _ThresholdScorer
    atom.scoring("cm")  # For special case


def test_delete_default():
    """Assert that the whole pipeline is deleted as default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.delete()
    assert not (atom.models or atom.metric)
    assert atom.results.empty


def test_delete_general_name():
    """Assert that the general name selects all models from that acronym."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR1", "LR2"])
    atom.delete("LR")
    assert not atom.models


def test_delete_general_number():
    """Assert that the general number selects all models with that number."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR0", "RF0"])
    atom.delete("0")
    assert not atom.models


def test_delete_duplicates():
    """Assert that duplicate models are ignored."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    atom.delete(["LR", "LR"])
    assert not atom.models


def test_delete_invalid_model():
    """Assert that an error is raised when model is not in pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    pytest.raises(ValueError, atom.delete, "GNB")


def test_delete_models_is_str():
    """Assert that for a string, a single model is deleted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree"])
    atom.delete("winner")
    assert atom.models == "Tree"
    assert atom.winner is atom.Tree
    assert len(atom.results) == 1
    assert not hasattr(atom, "LR")


def test_delete_models_is_sequence():
    """Assert that for a sequence, multiple models are deleted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree", "RF"])
    atom.delete(["Tree", "RF"])
    assert atom.models == "LR"
    assert atom.winner is atom.LR
    assert len(atom.results) == 1
