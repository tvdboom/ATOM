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
from atom.training import TrainerClassifier
from atom.utils import NotFittedError
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg


# Test utility properties =================================================== >>

def test_metric_property():
    """Assert that the metric property returns the metric names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="f1")
    assert atom.metric == "f1"


def test_metric_property_no_run():
    """Assert that the metric property doesn't crash when Trainer is not fit."""
    trainer = TrainerClassifier("lr", metric="r2", random_state=1)
    assert trainer.metric == "r2"


def test_models_property():
    """Assert that the models_ property returns the model subclasses."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["lr", "lda"])
    assert atom.models_ == [atom.LR, atom.LDA]


def test_results_property_sorted():
    """Assert that the results property returns sorted indices."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing("lr")
    atom.train_sizing("lda")
    assert list(atom.results.index.get_level_values("frac"))[:2] == [0.2, 0.2]


def test_results_property_reindex():
    """Assert that the results property is reindexed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(["lr", "lda", "knn", "rf"])
    assert list(atom.results.index.get_level_values("model"))[-4:] == atom.models


def test_results_property_dropna():
    """Assert that the results property doesn't return columns with NaNs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr")
    assert "mean_bagging" not in atom.results


def test_winner_property():
    """Assert that the winner property returns the best model in the pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["lr", "lda", "lgb"], n_calls=0)
    assert atom.winner.acronym == "LGB"


def test_shape_property():
    """Assert that the shape property returns the shape of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.shape == (len(X_bin), X_bin.shape[1] + 1)


def test_columns_property():
    """Assert that the columns property returns the columns of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert len(atom.columns) == len(X_bin.columns) + 1


def test_target_property():
    """Assert that the target property returns the last column in the dataset."""
    atom = ATOMClassifier(X_bin, "mean radius", random_state=1)
    assert atom.target == "mean radius"


def test_classes_property():
    """Assert that the classes property returns a df of the classes in y."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert list(atom.classes.index) == [0, 1, 2]


def test_n_classes_property():
    """Assert that the n_classes property returns the number of classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.n_classes == 3


# Data attributes =========================================================== >>

def test_dataset_property():
    """Assert that the dataset property returns the _data attribute."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dataset is atom._data


def test_train_property():
    """Assert that the train property returns the training set."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    assert atom.train.shape == (int((1 - 0.3) * len(X_bin)) + 1, X_bin.shape[1] + 1)


def test_test_property():
    """Assert that the test property returns the test set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.test.shape == (int(test_size * len(X_bin)), X_bin.shape[1] + 1)


def test_X_property():
    """Assert that the X property returns the feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.X.shape == (len(X_bin), X_bin.shape[1])


def test_y_property():
    """Assert that the y property returns the target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y.shape == (len(y_bin),)


def test_X_train_property():
    """Assert that the X_train property returns the training feature set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.X_train.shape == (int((1 - test_size) * len(X_bin)) + 1, X_bin.shape[1])


def test_X_test_property():
    """Assert that the X_test property returns the test feature set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.X_test.shape == (int(test_size * len(X_bin)), X_bin.shape[1])


def test_y_train_property():
    """Assert that the y_train property returns the training target column."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.y_train.shape == (int((1 - test_size) * len(X_bin)) + 1,)


def test_y_test_property():
    """Assert that the y_test property returns the training target column."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.y_test.shape == (int(test_size * len(X_bin)),)


# Test prediction methods =================================================== >>

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
    score = atom.score(X_bin, y_bin, sample_weight=atom.get_sample_weight("dataset"))
    assert isinstance(score, float)


# Test utility methods ====================================================== >>

def test_class_weights_invalid_dataset():
    """Assert that an error is raised if invalid value for dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.get_class_weight, "invalid")


@pytest.mark.parametrize("dataset", ["train", "test", "dataset"])
def test_class_weights_method(dataset):
    """Assert that the get_class_weight method returns a dict of the classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    class_weight = atom.get_class_weight(dataset)
    assert list(class_weight.keys()) == [0, 1, 2]


def test_sample_weights_invalid_dataset():
    """Assert that an error is raised if invalid value for dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(ValueError, atom.get_sample_weight, "invalid")


@pytest.mark.parametrize("dataset", ["train", "test", "dataset"])
def test_sample_weights_method(dataset):
    """Assert that the get_sample_weight method returns a list of the weights."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    sample_weight = atom.get_sample_weight(dataset)
    assert isinstance(sample_weight, list)


def test_calibrate_method():
    """Assert that the calibrate method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.calibrate)  # When not yet fitted
    atom.run("LR")
    atom.calibrate()
    assert atom.winner.estimator.__class__.__name__ == "CalibratedClassifierCV"


def test_not_fitted():
    """Assert that an error is raised when the class is not fitted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.scoring)


def test_invalid_metric():
    """Assert that an error is raised when an invalid metric_ is selected."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["ols", "br"])
    pytest.raises(ValueError, atom.scoring, metric="f1")


def test_metric_is_none():
    """Assert that it works for metric_=None."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["ols", "br"])
    atom.run("lgb", bagging=5)  # Test with and without bagging
    atom.scoring()
    assert 1 == 1  # Ran without errors


def test_metric_is_given():
    """Assert that it works for a specified metric_."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LDA", "PA"])
    atom.scoring("auc")
    atom.scoring("cm")
    assert 1 == 1  # Ran without errors


def test_models_is_all():
    """Assert that the whole pipeline is cleared for models="all"."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.clear("all")
    assert not (atom.models or atom.metric or atom.trainer)
    assert atom.results.empty


def test_invalid_model():
    """Assert that an error is raised when model is not in pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    pytest.raises(ValueError, atom.clear, "GNB")


def test_models_is_str():
    """Assert that a single model is cleared."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.clear("LDA")
    assert atom.models == ["LR"]
    assert atom.winner is atom.LR
    assert len(atom.results) == 1
    assert not hasattr(atom, "LDA")


def test_models_is_sequence():
    """Assert that multiple models are cleared."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA", "QDA"])
    atom.clear(["LDA", "QDA"])
    assert atom.models == ["LR"]
    assert atom.winner is atom.LR
    assert len(atom.results) == 1


def test_clear_successive_halving():
    """Assert that clearing works for successive_halving pipelines."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(["LR", "LDA", "QDA"], bagging=3)
    atom.clear(["LR"])
    assert "LR" not in atom.results.index.get_level_values(1)
    assert atom.winner is atom.LDA


def test_clear_train_sizing():
    """Assert that clearing works for successive_halving pipelines."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing(["LR", "LDA", "QDA"])
    atom.clear()
    assert not (atom.models or atom.metric or atom.trainer)
    assert atom.results.empty
