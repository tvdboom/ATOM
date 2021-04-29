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

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.utils import check_scaling
from .utils import X_bin, y_bin, X_class, y_class, X_reg, y_reg, X10_str, y10


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
    """Assert that all transformations are applied before predicting."""
    atom = ATOMClassifier(X10_str, y10, verbose=2, random_state=1)
    atom.encode(max_onehot=None)
    atom.prune(max_sigma=1.7)
    atom.run("Tree")
    pred_1 = atom.tree.predict(X10_str, pipeline=None)
    pred_2 = atom.tree.predict(X10_str, pipeline=True)
    assert not np.array_equal(pred_1, pred_2)

    # Fails because the data is not encoded
    pytest.raises(ValueError, atom.tree.predict, X10_str, pipeline=False)


def test_data_is_scaled():
    """Assert that the data is scaled for models that need it."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("SGD")
    assert sum(atom.sgd.predict(X_bin)) > 0  # Always 0 if not scaled


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
    print(atom.mnb.score_test)
    atom.mnb.reset_predictions()
    assert atom.mnb._pred_attrs[9] is None


def test_all_prediction_properties():
    """Assert that all prediction properties are saved as attributes when called."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "SGD"])
    assert isinstance(atom.lr.predict_train, np.ndarray)
    assert isinstance(atom.lr.predict_test, np.ndarray)
    assert isinstance(atom.lr.predict_proba_train, np.ndarray)
    assert isinstance(atom.lr.predict_proba_test, np.ndarray)
    assert isinstance(atom.lr.predict_log_proba_train, np.ndarray)
    assert isinstance(atom.lr.predict_log_proba_test, np.ndarray)
    assert isinstance(atom.lr.decision_function_train, np.ndarray)
    assert isinstance(atom.lr.decision_function_test, np.ndarray)
    assert isinstance(atom.lr.score_train, np.float64)
    assert isinstance(atom.lr.score_test, np.float64)


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


def test_y_train_property():
    """Assert that the y_train property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.y_train.equals(atom.mnb.y_train)
    assert atom.y_train.equals(atom.lr.y_train)


def test_y_test_property():
    """Assert that the y_test property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert atom.y_test.equals(atom.mnb.y_test)
    assert atom.y_test.equals(atom.lr.y_test)


def test_shape_property():
    """Assert that the shape property returns the shape of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom.lr.shape == atom.shape


def test_columns_property():
    """Assert that the columns property returns the columns of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert [i == j for i, j in zip(atom.lr.columns, atom.columns)]


def test_n_columns_property():
    """Assert that the n_columns property returns the number of columns."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom.lr.n_columns == atom.n_columns


def test_features_property():
    """Assert that the features property returns the features of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert [i == j for i, j in zip(atom.lr.features, atom.features)]


def test_n_features_property():
    """Assert that the n_features property returns the number of features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom.lr.n_features == atom.n_features


def test_target_property():
    """Assert that the target property returns the last column in the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert atom.lr.target == atom.target


# Test utility methods ============================================= >>

def test_delete():
    """Assert that models can be deleted."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("RF")
    atom.rf.delete()
    assert not atom.models
    assert not atom.metric


def test_scoring_invalid_dataset():
    """Assert that an error is raised when the dataset is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    pytest.raises(ValueError, atom.mnb.scoring, dataset="invalid")


def test_scoring_metric_None():
    """Assert that the scoring methods works when metric is empty."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    scores = atom.mnb.scoring()
    assert len(scores) == 7

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("MNB")
    scores = atom.mnb.scoring()
    assert len(scores) == 5

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("OLS")
    scores = atom.ols.scoring()
    assert len(scores) == 6


@patch("mlflow.tracking.MlflowClient.log_metric")
def test_rename_to_mlflow(mlflow):
    """Assert that renaming also changes the mlflow run."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    atom.scoring()
    assert mlflow.call_count == 8  # 7 from scoring + 1 from training
