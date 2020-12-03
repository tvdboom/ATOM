# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for basepredictor.py

"""

# Standard packages
import pytest
import numpy as np
import pandas as pd

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
    assert list(atom._branches.keys()) == ["main"]


def test_delattr_normal():
    """Assert that trainer attributes can be deleted normally."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    del atom.models
    assert not hasattr(atom, "models")


# Test utility properties ========================================== >>

def test_branch_property():
    """Assert that the branch property returns the current branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom.branch, Branch)


def test_pipeline_property():
    """Assert that the pipeline property returns the estimators in the branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom.pipeline, pd.Series)


def test_feature_importance_property():
    """Assert that the feature_importance returns the list of features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection('univariate', n_features=10)
    assert isinstance(atom.feature_importance, list)


def test_metric_property():
    """Assert that the metric property returns the metric names."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric="f1")
    assert atom.metric == "f1"


def test_metric_property_no_run():
    """Assert that the metric property doesn't crash when Trainer is not fit."""
    trainer = DirectClassifier("lr", metric="r2", random_state=1)
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


# Test data attributes ============================================= >>

def test_dataset_property():
    """Assert that the dataset property returns the data in the branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.dataset.equals(atom.branch.data)


def test_train_property():
    """Assert that the train property returns the training set."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    assert atom.train.equals(atom.branch.train)


def test_test_property():
    """Assert that the test property returns the test set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.test.equals(atom.branch.test)


def test_X_property():
    """Assert that the X property returns the feature set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.X.equals(atom.branch.X)


def test_y_property():
    """Assert that the y property returns the target column."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.y.equals(atom.branch.y)


def test_X_train_property():
    """Assert that the X_train property returns the training feature set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.X_train.equals(atom.branch.X_train)


def test_X_test_property():
    """Assert that the X_test property returns the test feature set."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.X_test.equals(atom.branch.X_test)


def test_y_train_property():
    """Assert that the y_train property returns the training target column."""
    test_size = 0.3
    atom = ATOMClassifier(X_bin, y_bin, test_size=test_size, random_state=1)
    assert atom.y_train.equals(atom.branch.y_train)


def test_y_test_property():
    """Assert that the y_test property returns the training target column."""
    atom = ATOMClassifier(X_bin, y_bin, test_size=0.3, random_state=1)
    assert atom.y_test.equals(atom.branch.y_test)


def test_shape_property():
    """Assert that the shape property returns the shape of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.shape == atom.branch.shape


def test_columns_property():
    """Assert that the columns property returns the columns of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.columns == atom.branch.columns


def test_features_property():
    """Assert that the features property returns the features of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom.features == atom.branch.features


def test_target_property():
    """Assert that the target property returns the last column in the dataset."""
    atom = ATOMClassifier(X_bin, "mean radius", random_state=1)
    assert atom.target == atom.branch.target


def test_classes_property():
    """Assert that the classes property returns a df of the classes in y."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.classes.equals(atom.branch.classes)


def test_n_classes_property():
    """Assert that the n_classes property returns the number of classes."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    assert atom.n_classes == atom.branch.n_classes


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
    """Assert that the whole pipeline is deleted for models="all"."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.delete("all")
    assert not (atom.models or atom.metric or atom._approach)
    assert atom.results.empty


def test_invalid_model():
    """Assert that an error is raised when model is not in pipeline."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    pytest.raises(ValueError, atom.delete, "GNB")


def test_models_is_str():
    """Assert that a single model is deleted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"])
    atom.delete("lda")
    assert atom.models == ["LR"]
    assert atom.winner is atom.LR
    assert len(atom.results) == 1
    assert not hasattr(atom, "LDA")


def test_models_is_sequence():
    """Assert that multiple models are deleted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA", "QDA"])
    atom.delete(["LDA", "QDA"])
    assert atom.models == ["LR"]
    assert atom.winner is atom.LR
    assert len(atom.results) == 1


def test_delete_successive_halving():
    """Assert that deleting works for successive halving pipelines."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(["LR", "LDA", "QDA"], bagging=3)
    atom.delete(["LR"])
    assert "LR" not in atom.results.index.get_level_values(1)
    assert atom.winner is atom.LDA


def test_delete_train_sizing():
    """Assert that deleting works for train sizing pipelines."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.train_sizing(["LR", "LDA", "QDA"])
    atom.delete()
    assert not (atom.models or atom.metric or atom._approach)
    assert atom.results.empty
