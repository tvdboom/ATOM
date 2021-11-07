# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for basemodel.py

"""

# Standard packages
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sklearn.metrics import accuracy_score, r2_score, recall_score

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.utils import check_scaling
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg, X10_str, y10


# Test magic methods =============================================== >>

def test_getattr():
    """Assert that branch attributes can be called from a model."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.balance(strategy="smote")
    atom.run("Tree")
    assert isinstance(atom.tree.shape, tuple)
    assert isinstance(atom.tree.alcohol, pd.Series)
    assert isinstance(atom.tree.head(), pd.DataFrame)
    with pytest.raises(AttributeError, match=r".*has no attribute.*"):
        print(atom.tree.data)


def test_contains():
    """Assert that we can test if model contains a column."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("Tree")
    assert "alcohol" in atom.tree


def test_getitem():
    """Assert that the models are subscriptable."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("Tree")
    assert atom.tree["alcohol"].equals(atom.dataset["alcohol"])
    with pytest.raises(TypeError, match=r".*subscriptable with type str.*"):
        print(atom.tree[2])


# Test utility properties ========================================== >>

def test_results_property():
    """Assert that an error is raised when the model doesn't have the method."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.results, pd.Series)


# Test prediction methods ========================================== >>

def test_invalid_method():
    """Assert that an error is raised when the model doesn't have the method."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("SGD")
    pytest.raises(AttributeError, atom.sgd.predict_proba, X_bin)


def test_transformations_first():
    """Assert that the transformations are applied before predicting."""
    atom = ATOMClassifier(X10_str, y10, verbose=2, random_state=1)
    atom.encode(max_onehot=None)
    atom.prune(max_sigma=1.7)
    atom.run("Tree")
    assert not atom.errors


def test_data_is_scaled():
    """Assert that the data is scaled for models that need it."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("SGD")
    assert sum(atom.sgd.predict(X_bin)) > 0  # Always 0 if not scaled


def test_score_metric_is_None():
    """Assert that the score returns accuracy for classification tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    accuracy = accuracy_score(y_bin, atom.predict(X_bin))
    assert atom.tree.score(X_bin, y_bin) == accuracy


def test_score_regression():
    """Assert that the score returns r2 for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    r2 = r2_score(y_reg, atom.predict(X_reg))
    assert atom.tree.score(X_reg, y_reg) == r2


def test_score_custom_metric():
    """Assert that the score method works when sample weights are provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    recall = recall_score(y_bin, atom.predict(X_bin))
    assert atom.tree.score(X_bin, y_bin, metric="recall") == recall


def test_score_with_sample_weights():
    """Assert that the score method works when sample weights are provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    score = atom.tree.score(X_bin, y_bin, sample_weight=list(range(len(y_bin))))
    assert isinstance(score, np.float64)


# Test properties ================================================== >>

def test_reset_predictions():
    """Assert that reset_predictions removes the made predictions."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.plot_roc()
    assert isinstance(atom.mnb._pred[4], np.ndarray)  # Predictions are made
    atom.mnb.reset_predictions()
    assert atom.mnb._pred[4] is None  # Predictions are reset


@pytest.mark.parametrize("dataset", ["train", "test", "holdout"])
def test_all_prediction_properties(dataset):
    """Assert that all prediction properties can be called."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run(["LR", "SGD"])
    assert isinstance(getattr(atom.lr, f"predict_{dataset}"), np.ndarray)
    assert isinstance(getattr(atom.lr, f"predict_proba_{dataset}"), np.ndarray)
    assert isinstance(getattr(atom.lr, f"predict_log_proba_{dataset}"), np.ndarray)
    assert isinstance(getattr(atom.lr, f"decision_function_{dataset}"), np.ndarray)
    assert isinstance(getattr(atom.lr, f"score_{dataset}"), np.float64)


def test_dataset_property():
    """Assert that the dataset property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.dataset.equals(atom.mnb.dataset)
    assert check_scaling(atom.lr.dataset)


def test_train_property():
    """Assert that the train property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.train.equals(atom.mnb.train)
    assert check_scaling(atom.lr.train)


def test_test_property():
    """Assert that the test property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.test.equals(atom.mnb.test)
    assert check_scaling(atom.lr.test)


def test_holdout_property():
    """Assert that the holdout property is calculated."""
    atom = ATOMClassifier(X10_str, y10, holdout_size=0.1, random_state=1)
    atom.encode()
    atom.run("MNB")
    assert not atom.holdout.equals(atom.mnb.holdout)
    assert len(atom.mnb.holdout.columns) > 3  # Holdout is transformed


def test_X_property():
    """Assert that the X property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.X.equals(atom.mnb.X)
    assert check_scaling(atom.lr.X)


def test_y_property():
    """Assert that the y property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.y.equals(atom.mnb.y)
    assert atom.y.equals(atom.lr.y)


def test_X_train_property():
    """Assert that the X_train property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.X_train.equals(atom.mnb.X_train)
    assert check_scaling(atom.lr.X_train)


def test_X_test_property():
    """Assert that the X_test property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.X_test.equals(atom.mnb.X_test)
    assert check_scaling(atom.lr.X_test)


def test_X_holdout_property():
    """Assert that the X_holdout property is calculated."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    assert atom.mnb.X_holdout.equals(atom.mnb.holdout.iloc[:, :-1])


def test_y_train_property():
    """Assert that the y_train property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.y_train.equals(atom.mnb.y_train)
    assert atom.y_train.equals(atom.lr.y_train)


def test_y_holdout_property():
    """Assert that the y_holdout property is calculated."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    assert atom.mnb.y_holdout.equals(atom.mnb.holdout.iloc[:, -1])


# Test utility methods ============================================= >>

def test_delete():
    """Assert that models can be deleted."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("RF")
    atom.rf.delete()
    assert not atom.models
    assert not atom.metric


def test_evaluate_invalid_dataset():
    """Assert that an error is raised when the dataset is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    with pytest.raises(ValueError, match=r".*Unknown value for the dataset.*"):
        atom.mnb.evaluate(dataset="invalid")


def test_evaluate_no_holdout():
    """Assert that an error is raised when there's no holdout set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    with pytest.raises(ValueError, match=r".*No holdout data set.*"):
        atom.mnb.evaluate(dataset="holdout")


@pytest.mark.parametrize("dataset", ["train", "test", "holdout"])
def test_evaluate_metric_None(dataset):
    """Assert that the evaluate method works when metric is empty."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    scores = atom.mnb.evaluate(dataset=dataset)
    assert len(scores) == 9

    atom = ATOMClassifier(X_class, y_class, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    scores = atom.mnb.evaluate(dataset=dataset)
    assert len(scores) == 6

    atom = ATOMRegressor(X_reg, y_reg, holdout_size=0.1, random_state=1)
    atom.run("OLS")
    scores = atom.ols.evaluate(dataset=dataset)
    assert len(scores) == 7


@patch("mlflow.tracking.MlflowClient.log_metric")
def test_rename_to_mlflow(mlflow):
    """Assert that renaming also changes the mlflow run."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    atom.evaluate()
    assert mlflow.call_count == 11  # 9 from evaluate + 2 from training
