# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for basemodel.py

"""

# Standard packages
import glob
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from skopt.learning import GaussianProcessRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, r2_score, recall_score

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.training import DirectClassifier
from atom.utils import check_scaling
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg, bin_train,
    bin_test, X10_str, y10,
)


# Test __init__ and magic methods ================================== >>

def test_scaler():
    """Assert that a scaler is made for models that need scaling."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LGB", "LDA"])
    assert atom.lgb.scaler and not atom.lda.scaler


def test_repr():
    """Assert that the __repr__ method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LDA")
    assert str(atom.lda).startswith("Linear Discriminant")


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
    assert isinstance(atom.tree[["alcohol", "ash"]], pd.DataFrame)
    with pytest.raises(TypeError, match=r".*subscriptable with types.*"):
        print(atom.tree[2])


# Test training ==================================================== >>

def test_n_calls_lower_n_initial_points():
    """Assert than an error is raised when n_calls<n_initial_points."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LDA"], n_calls=(5, 2), n_initial_points=(2, 3))
    assert atom.errors.get("LDA")


def test_est_params_removed_from_bo():
    """Assert that all params in est_params are dropped from the BO."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", n_calls=5, est_params={"n_estimators": 220})
    assert "n_estimators" not in atom.lgb.bo.params[0]


def test_no_hyperparameters_left():
    """Assert that the BO is skipped when there are no hyperparameters."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models="BNB", n_calls=10, est_params={"alpha": 1.0, "fit_prior": True})
    assert atom.bnb.bo.empty


def test_est_params_unknown_param():
    """Assert that unknown parameters in est_params are caught."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], n_calls=5, est_params={"test": 220})
    assert list(atom.errors.keys()) == ["LR"]  # LGB passes since it accepts kwargs


def test_est_params_unknown_param_fit():
    """Assert that unknown parameters in est_params_fit are caught."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(RuntimeError, atom.run, ["LR", "LGB"], est_params={"test_fit": 220})


def test_est_params_default_method():
    """Assert that custom parameters overwrite the default ones."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("RF", est_params={"n_jobs": 3})
    assert atom.rf.estimator.get_params()["n_jobs"] == 3
    assert atom.rf.estimator.get_params()["random_state"] == 1


@pytest.mark.parametrize("est", ["GP", "ET", "RF", "GBRT", GaussianProcessRegressor()])
def test_all_base_estimators(est):
    """Assert that the pipeline works for all base estimators."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_calls=5, bo_params={"base_estimator": est})


def test_sample_weights_fit():
    """Assert that sample weights can be used with the BO."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LGB",
        n_calls=5,
        est_params={"sample_weight_fit": list(range(len(atom.y_train)))},
    )


@pytest.mark.parametrize("model", ["XGB", "LGB", "CatB"])
def test_early_stopping(model):
    """Assert than early stopping works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(model, n_calls=5, bo_params={"early_stopping": 0.1})
    assert getattr(atom, model)._stopped != ("---", "---")


@pytest.mark.parametrize("model", ["XGB", "LGB", "CatB"])
def test_est_params_for_fit(model):
    """Assert that est_params is used for fit if ends in _fit."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(model, est_params={"early_stopping_rounds_fit": 2})
    assert getattr(atom, model)._stopped != ("---", "---")


def test_skip_duplicate_calls():
    """Assert that calls with the same parameters skip the calculation."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("dummy", n_calls=5)
    assert atom.dummy.bo["score"].nunique() < len(atom.dummy.bo["score"])


@patch("mlflow.set_tag")
def test_nested_runs_to_mlflow(mlflow):
    """Assert that the BO is logged to mlflow as nested runs."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.log_bo = True
    atom.run("Tree", n_calls=5)
    assert mlflow.call_count == 5  # Only called at iterations


@pytest.mark.parametrize("cv", [1, 3])
def test_verbose_is_1(cv):
    """Assert that the pipeline works for verbose=1."""
    atom = ATOMRegressor(X_reg, y_reg, verbose=1, random_state=1)
    atom.run("RF", n_calls=5, bo_params={"cv": cv})
    assert not atom.errors


@patch("mlflow.set_tags")
def test_run_set_tags_to_mlflow(mlflow):
    """Assert that the mlflow run gets tagged."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    mlflow.assert_called_with(
        {
            "fullname": atom.gnb.fullname,
            "branch": atom.gnb.branch.name,
            "time": atom.gnb.time_fit,
        }
    )


@patch("mlflow.log_params")
def test_run_log_params_to_mlflow(mlflow):
    """Assert that model parameters are logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    assert mlflow.call_count == 1


@patch("mlflow.log_metric")
def test_run_log_metric_to_mlflow(mlflow):
    """Assert that metrics are logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB", metric=["f1", "recall", "accuracy"])
    assert mlflow.call_count == 6


@patch("mlflow.log_metric")
def test_run_log_evals_to_mlflow(mlflow):
    """Assert that eval metrics are logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("CatB")
    assert mlflow.call_count > 10


@patch("mlflow.sklearn.log_model")
def test_run_log_models_to_mlflow(mlflow):
    """Assert that models are logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.log_model = True
    atom.run("LGB")
    mlflow.assert_called_with(atom.lgb.estimator, "LGBMClassifier")


@patch("mlflow.log_artifact")
def test_run_log_data_to_mlflow(mlflow):
    """Assert that train and test sets are logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.log_data = True
    atom.run("GNB")
    assert mlflow.call_count == 2  # Train and test set


@patch("mlflow.sklearn.log_model")
def test_run_log_pipeline_to_mlflow(mlflow):
    """Assert that renaming also changes the mlflow run."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.log_pipeline = True
    atom.run("GNB")
    assert mlflow.call_count == 2  # Model + Pipeline


def test_bootstrap_attribute_types():
    """Assert that the bootstrap attributes have python types (not numpy)."""
    # For single-metric
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", n_calls=5, n_bootstrap=5)
    assert isinstance(atom.lgb.metric_bootstrap, np.ndarray)
    assert isinstance(atom.lgb.mean_bootstrap, float)

    # For multi-metric
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", metric=("f1", "auc", "recall"), n_bootstrap=5)
    assert isinstance(atom.lgb.metric_bootstrap, np.ndarray)
    assert isinstance(atom.lgb.mean_bootstrap, list)


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
    atom.run("Tree")
    assert isinstance(atom.tree.predict(X10_str), np.ndarray)


def test_data_is_scaled():
    """Assert that the data is scaled for models that need it."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("SGD")
    assert sum(atom.sgd.predict(X_bin)) > 0  # Always 0 if not scaled


def test_score_metric_is_None():
    """Assert that the score returns accuracy for classification tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    accuracy = atom.tree.score(X_bin, y_bin)
    assert accuracy == accuracy_score(y_bin, atom.predict(X_bin))


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


# Test prediction properties ======================================= >>

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

def test_calibrate_invalid_task():
    """Assert than an error is raised when task="regression"."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("OLS")
    pytest.raises(PermissionError, atom.ols.calibrate)


def test_calibrate():
    """Assert that calibrate method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.calibrate(cv=3)
    assert isinstance(atom.mnb.estimator, CalibratedClassifierCV)


def test_calibrate_prefit():
    """Assert that calibrate method works as intended when cv="prefit"."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.calibrate(cv="prefit")
    assert isinstance(atom.mnb.estimator, CalibratedClassifierCV)


def test_calibrate_reset_predictions():
    """Assert that the prediction attributes are reset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.calibrate()
    assert atom.mnb._pred == [None] * 15


@patch("mlflow.sklearn.log_model")
def test_calibrate_to_mlflow(mlflow):
    """Assert that the CCV is logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    atom.gnb.calibrate()
    mlflow.assert_called_with(atom.gnb.estimator, "CalibratedClassifierCV")


def test_cross_validate():
    """Assert that the cross_validate method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert isinstance(atom.lr.cross_validate(), dict)
    assert isinstance(atom.lr.cross_validate(scoring="AP"), dict)


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


def test_export_pipeline():
    """Assert that the pipeline can be retrieved from the model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.run("LR")
    assert len(atom.lr.export_pipeline(verbose=2)) == 3


@patch("tempfile.gettempdir")
def test_export_pipeline_memory(func):
    """Assert that memory is True triggers a temp dir."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    atom.run("LR")
    atom.lr.export_pipeline(memory=True)
    func.assert_called_once()


def test_full_train():
    """Assert that the full_train method trains on the test set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB")
    atom.lgb.full_train()
    assert atom.lgb.score_test == 1.0  # Perfect score on test


def test_full_train_holdout():
    """Assert that the full_train method trains on the holdout set."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.2, random_state=1)
    atom.run("Tree")
    atom.tree.full_train(include_holdout=True)
    assert atom.tree.score_holdout == 1.0  # Perfect score on holdout


def test_full_train_reset_predictions():
    """Assert that the prediction attributes are reset."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.full_train()
    assert atom.mnb._pred == [None] * 15


@patch("mlflow.sklearn.log_model")
def test_full_train_to_mlflow(mlflow):
    """Assert that the full trained estimator is logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    atom.gnb.full_train()
    mlflow.assert_called_with(atom.gnb.estimator, "full_train_GaussianNB")


def test_rename():
    """Assert that the model's tag can be changed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "MNB_2"])
    pytest.raises(PermissionError, atom.mnb.rename, name="_2")
    atom.mnb.rename("_3")
    assert atom.models == ["MNB_3", "MNB_2"]
    atom.mnb_2.rename()
    assert atom.models == ["MNB_3", "MNB"]


@patch("mlflow.tracking.MlflowClient.set_tag")
def test_rename_to_mlflow(mlflow):
    """Assert that renaming also changes the mlflow run."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    atom.gnb.rename("GNB2")
    mlflow.assert_called_with(atom.gnb2._run.info.run_id, "mlflow.runName", "GNB2")


def test_save_estimator():
    """Assert that the save_estimator saves a pickle file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.save_estimator(FILE_DIR + "auto")
    assert glob.glob(FILE_DIR + "MultinomialNB")


def test_transform():
    """Assert that new data can be transformed by the model's pipeline."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    atom.run("LR")
    X = atom.lr.transform(X10_str)
    assert len(X.columns) > 3  # Data is one-hot encoded
    assert check_scaling(X)  # Data is scaled
