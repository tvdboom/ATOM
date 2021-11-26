# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for basepredictor.py

"""

# Standard packages
import pytest
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.branch import Branch
from atom.training import DirectClassifier
from atom.utils import NotFittedError
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg, bin_train, bin_test


# Test magic methods =============================================== >>

def test_getattr_branch():
    """Assert that branches can be called from the trainer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    assert atom.b2 is atom._branches["b2"]


def test_getattr_attr_from_branch():
    """Assert that branch attributes can be called from the trainer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.pipeline is atom.branch.pipeline


def test_getattr_model():
    """Assert that the models can be called as attributes from the trainer."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom.tree is atom._models[0]


def test_getattr_column():
    """Assert that the columns can be accessed as attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.apply(lambda x: np.log(x["mean radius"]), columns="log_column")
    assert isinstance(atom.log_column, pd.Series)


def test_getattr_dataframe():
    """Assert that the dataset attributes can be called from atom."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom.head(), pd.DataFrame)


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


def test_delattr_branch():
    """Assert that branches can be deleted through del."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.branch = "b2"
    atom.branch = "b3"
    del atom.branch
    assert list(atom._branches) == ["og", "master", "b2"]
    del atom.b2
    assert list(atom._branches) == ["og", "master"]


def test_delattr_models():
    """Assert that models can be deleted through del."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    del atom.winner
    assert atom.models == "MNB"
    del atom.winner
    assert not atom.models


def test_delattr_normal():
    """Assert that trainer attributes can be deleted normally."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    del atom._models
    assert not hasattr(atom, "_models")


def test_delattr_invalid():
    """Assert that an error is raised when there is no such attribute."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(AttributeError, match=r".*object has no attribute.*"):
        del atom.invalid


def test_contains():
    """Assert that we can test if a trainer contains a column."""
    trainer = DirectClassifier(models="LR", random_state=1)
    assert "mean radius" not in trainer
    trainer.run(bin_train, bin_test)
    assert "mean radius" in trainer


def test_len():
    """Assert that the length of a trainer is the length of the dataset."""
    trainer = DirectClassifier(models="LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert len(trainer) == len(X_bin)


def test_getitem():
    """Assert that atom is subscriptable."""
    trainer = DirectClassifier(models="LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer["LR"] is trainer["lr"] is trainer.lr
    assert trainer["mean radius"] is trainer.dataset["mean radius"]
    assert isinstance(trainer[["mean radius", "mean texture"]], pd.DataFrame)
    with pytest.raises(ValueError, match=r".*has no model or column.*"):
        print(trainer["invalid"])
    with pytest.raises(TypeError, match=r".*subscriptable with types.*"):
        print(trainer[2.3])


# Test utility properties ========================================== >>

def test_branch_property():
    """Assert that the branch property returns the current branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom.branch, Branch)


def test_models_property():
    """Assert that the models property returns the model names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree"])
    assert atom.models == ["LR", "Tree"]


def test_models_property_no_run():
    """Assert that the models property doesn't crash for unfitted trainers."""
    trainer = DirectClassifier(["LR", "Tree"], metric="r2", random_state=1)
    assert trainer.models == ["LR", "Tree"]


def test_metric_property():
    """Assert that the metric property returns the metric names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="f1")
    assert atom.metric == "f1"


def test_metric_property_no_run():
    """Assert that the metric property doesn't crash for unfitted trainers."""
    trainer = DirectClassifier("lr", metric="r2", random_state=1)
    assert trainer.metric == "r2"


def test_errors_property():
    """Assert that the errors property returns the model's errors."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"], n_calls=5, n_initial_points=(2, 6))
    assert "LGB" in atom.errors


def test_winner_property():
    """Assert that the winner property returns the best model in the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "Tree", "LGB"], n_calls=0)
    assert atom.winner is atom.lgb


def test_results_property():
    """Assert that the results property returns an overview of the results."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom.results.shape == (1, 4)


def test_results_property_dropna():
    """Assert that the results property doesn't return columns with NaNs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert "mean_bootstrap" not in atom.results


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


# Test prediction methods ========================================== >>

def test_reset_predictions():
    """Assert that we can reset all predictions."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    print(atom.lr.predict_proba_train)
    print(atom.lgb.predict_test)
    atom.reset_predictions()
    assert atom.lr._pred == [None] * 15


def test_predict():
    """Assert that the predict method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.predict, X_bin)  # When not yet fitted
    atom.run("LR")
    assert isinstance(atom.predict(X_bin), np.ndarray)


def test_predict_proba():
    """Assert that the predict_proba method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.predict_proba, X_bin)
    atom.run("LR")
    assert isinstance(atom.predict_proba(X_bin), np.ndarray)


def test_predict_log_proba():
    """Assert that the predict_log_proba method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.predict_log_proba, X_bin)
    atom.run("LR")
    assert isinstance(atom.predict_log_proba(X_bin), np.ndarray)


def test_decision_function():
    """Assert that the decision_function method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.decision_function, X_bin)
    atom.run("LR")
    assert isinstance(atom.decision_function(X_bin), np.ndarray)


def test_score():
    """Assert that the score method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.score, X_bin, y_bin)
    atom.run("LR")
    assert isinstance(atom.score(X_bin, y_bin), float)


def test_score_sample_weights():
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
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.successive_halving(["OLS", "ET", "RF", "LGB"])
    assert atom._get_model_name("2") == ["OLS2", "ET2"]


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


def test_available_models():
    """Assert that the available_models method shows the models per task."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    models = atom.available_models()
    assert isinstance(models, pd.DataFrame)
    assert "LR" in models["acronym"].unique()
    assert "BR" not in models["acronym"].unique()


def test_delete_default():
    """Assert that all models in branch are deleted as default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.delete()  # No models
    atom.run(["LR", "LDA"])
    atom.delete()  # All models
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


@pytest.mark.parametrize("metric", ["ap", "roc_auc_ovo", "f1"])
def test_evaluate(metric):
    """Assert that the evaluate method works when metric is None."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.evaluate)
    atom.run(["Tree", "PA"])
    assert isinstance(atom.evaluate(metric=metric), pd.DataFrame)


def test_class_weights_invalid_dataset():
    """Assert that an error is raised if invalid value for dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.get_class_weight, "invalid")


def test_get_class_weights_regression():
    """Assert that an error is raised when called from regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(PermissionError, atom.get_class_weight)


@pytest.mark.parametrize("dataset", ["train", "test", "dataset"])
def test_get_class_weights(dataset):
    """Assert that the get_class_weight method returns a dict of the classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    class_weight = atom.get_class_weight(dataset)
    assert list(class_weight) == [0, 1, 2]


def test_stacking():
    """Assert that the stacking method creates a Stack model."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    pytest.raises(NotFittedError, atom.stacking)
    atom.run(["LR", "LGB"])
    atom.stacking()
    assert hasattr(atom, "stack")
    assert "Stack" in atom.models
    assert atom.stack._run


def test_stacking_invalid_models():
    """Assert that an error is raised when <2 models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    with pytest.raises(ValueError, match=r".*contain at least two.*"):
        atom.stacking()


def test_stacking_custom_models():
    """Assert that stacking can be created selecting the models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.stacking)
    atom.run(["LR", "LDA", "LGB"])
    atom.stacking(models=["LDA", "LGB"])
    assert list(atom.stack._models) == ["LDA", "LGB"]


def test_stacking_models_from_branch():
    """Assert that only the models from the current branch are passed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    atom.branch = "b2"
    atom.balance()
    atom.run(["RF", "ET"])
    atom.stacking()
    assert list(atom.stack._models) == ["RF", "ET"]


def test_stacking_different_name():
    """Assert that the acronym is added in front of the new name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    atom.stacking(name="stack_1")
    atom.stacking(name="_2")
    assert hasattr(atom, "Stack_1") and hasattr(atom, "Stack_2")


def test_stacking_unknown_predefined_final_estimator():
    """Assert that an error is raised when the final estimator is unknown."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    with pytest.raises(ValueError, match=r".*Unknown model.*"):
        atom.stacking(final_estimator="invalid")


def test_stacking_invalid_predefined_final_estimator():
    """Assert that an error is raised when the final estimator is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    with pytest.raises(ValueError, match=r".*can not perform.*"):
        atom.stacking(final_estimator="OLS")


def test_stacking_predefined_final_estimator():
    """Assert that the final estimator accepts predefined models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"])
    atom.stacking(final_estimator="LDA")
    assert isinstance(atom.stack.estimator.final_estimator_, LDA)


def test_voting():
    """Assert that the voting method creates a Vote model."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    pytest.raises(NotFittedError, atom.voting)
    atom.run(["LR", "LGB"])
    atom.voting(name="2")
    assert hasattr(atom, "Vote2")
    assert "Vote2" in atom.models
    assert atom.vote2._run


def test_voting_invalid_models():
    """Assert that an error is raised when <2 models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    with pytest.raises(ValueError, match=r".*contain at least two.*"):
        atom.voting()
