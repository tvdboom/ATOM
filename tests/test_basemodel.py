# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for basemodel.py

"""

import glob
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, r2_score, recall_score
from skopt.learning import GaussianProcessRegressor
from skopt.space.space import Integer

from atom import ATOMClassifier, ATOMRegressor
from atom.utils import check_scaling

from .conftest import (
    X10_str, X_bin, X_class, X_idx, X_reg, y10, y_bin, y_class, y_idx, y_reg,
)


# Test magic methods ================================== >>

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
    atom.run("LGB", n_calls=2, n_initial_points=1, est_params={"n_estimators": 220})
    assert "n_estimators" not in atom.lgb.bo.params[0]


def test_bo_with_no_hyperparameters():
    """Assert that the BO is skipped when there are no hyperparameters."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models="BNB", n_calls=10, est_params={"alpha": 1.0, "fit_prior": True})
    assert atom.bnb.bo.empty


def test_custom_dimensions_is_name():
    """Assert that the parameters to tune can be set by name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"dimensions": "max_iter"},
    )
    assert list(atom.lr.best_params) == ["max_iter"]


def test_custom_dimensions_is_name_excluded():
    """Assert that the parameters to tune can be excluded by name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="CNB",
        n_calls=2,
        n_initial_points=2,
        bo_params={"dimensions": "!fit_prior"},
    )
    assert list(atom.cnb.best_params) == ["alpha", "norm"]


def test_custom_dimensions_name_is_invalid():
    """Assert that an error is raised when an invalid parameter is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=r".*is not a predefined hyperparameter.*"):
        atom.run("LR", n_calls=5, bo_params={"dimensions": "invalid"})


def test_custom_dimensions_is_dim():
    """Assert that the custom dimensions are for all models if dimension."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"dimensions": Integer(10, 20, name="max_iter")},
        random_state=1,
    )
    assert list(atom.lr.best_params) == ["max_iter"]


def test_custom_dimensions_include_and_excluded():
    """Assert that an error is raised when parameters are included and excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=r".*either include or exclude.*"):
        atom.run("LR", n_calls=5, bo_params={"dimensions": ["!max_iter", "penalty"]})


def test_default_parameters():
    """Assert that default parameters are used when n_intial_points=1."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models="MLP", n_calls=2, n_initial_points=1)
    assert atom.mlp.bo.params[0]["hidden_layer_sizes"] == (100,)
    assert atom.mlp.bo.params[0]["solver"] == "adam"


def test_default_parameter_not_in_dimension():
    """Assert that a random value is assigned for a parameter outside the space."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="SGD",
        n_calls=2,
        n_initial_points=1,
        est_params={"learning_rate": "constant"},
    )
    assert atom.sgd.bo.params[0]["eta0"] != 0  # Is not default


@pytest.mark.parametrize("est", ["GP", "ET", "RF", "GBRT", GaussianProcessRegressor()])
def test_all_base_estimators(est):
    """Assert that the pipeline works for all base estimators."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_calls=5, bo_params={"base_estimator": est})


def test_sample_weight_fit():
    """Assert that sample weights can be used with the BO."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LGB",
        n_calls=5,
        est_params={"sample_weight_fit": list(range(len(atom.y_train)))},
    )


def test_bo_with_pipeline():
    """Assert that the BO works with a transformer pipeline."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    atom.run("SGD", n_calls=5, n_initial_points=2)
    assert not atom.sgd.bo.empty


@pytest.mark.parametrize("model", ["XGB", "LGB", "CatB"])
def test_early_stopping(model):
    """Assert than early stopping works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models=model,
        n_calls=5,
        est_params={"n_estimators": 10},
        bo_params={"early_stopping": 0.1},
    )
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
    assert mlflow.call_count == 6  # BO iterations + fit


@pytest.mark.parametrize("cv", [1, 3])
def test_verbose_is_1(cv):
    """Assert that the pipeline works for verbose=1."""
    atom = ATOMRegressor(X_reg, y_reg, verbose=1, random_state=1)
    atom.run("Tree", n_calls=5, bo_params={"cv": cv})
    assert not atom.errors


@patch("mlflow.set_tags")
def test_run_set_tags_to_mlflow(mlflow):
    """Assert that the mlflow run gets tagged."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    mlflow.assert_called_once()


@patch("mlflow.log_params")
def test_run_log_params_to_mlflow(mlflow):
    """Assert that model parameters are logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    assert mlflow.call_count == 1


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
    """Assert that the property returns an overview of the training."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.results, pd.Series)


def test_metrics_property_single_metric():
    """Assert that the metric properties return a value for single metric."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric="f1")
    assert isinstance(atom.tree.metric_train, float)
    assert isinstance(atom.tree.metric_test, float)


def test_metrics_property_multi_metric():
    """Assert that the metric properties return a list for multi-metric."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric=["f1", "recall"])
    assert isinstance(atom.tree.metric_train, list)
    assert isinstance(atom.tree.metric_test, list)


# Test prediction methods ========================================== >>

def test_invalid_method():
    """Assert that an error is raised when the model doesn't have the method."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("SGD")
    pytest.raises(AttributeError, atom.sgd.predict_proba, X_bin)


def test_predictions_from_index():
    """Assert that predictions when providing indices."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, holdout_size=0.1, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.predict_proba("index_4"), np.ndarray)
    assert isinstance(atom.tree.predict(["index_4", "index_8"]), np.ndarray)
    assert isinstance(atom.tree.predict_log_proba(atom.holdout.index[0]), np.ndarray)


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


def test_score_with_sample_weight():
    """Assert that the score method works when sample weights are provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    score = atom.tree.score(X_bin, y_bin, sample_weight=list(range(len(y_bin))))
    assert isinstance(score, np.float64)


# Test prediction properties ======================================= >>

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
    atom = ATOMClassifier(X10_str, y10, holdout_size=0.3, random_state=1)
    atom.encode()
    atom.run(["LR", "Tree"])
    assert not atom.lr.holdout.equals(atom.tree.holdout)  # Scaler vs no scaler
    assert len(atom.lr.holdout.columns) > 3  # Holdout is transformed


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


def test_calibrate_clear():
    """Assert that the clear method is called."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    print(atom.tree.predict_log_proba_test)
    assert atom.tree._pred[7] is not None
    atom.tree.calibrate()
    assert atom.tree._pred[7] is None


def test_calibrate_new_mlflow_run():
    """Assert that a new mlflow run is created."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    run = atom.gnb._run
    atom.gnb.calibrate()
    assert atom.gnb._run is not run


def test_clear():
    """Assert that the clear method resets the model's attributes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    atom.beeswarm_plot(display=False)
    assert atom.lr._pred[3] is not None
    assert atom.lr._scores["train"]
    assert not atom.lr._shap._shap_values.empty
    atom.clear()
    assert atom.lr._pred == [None] * 15
    assert not atom.lr._scores["train"]
    assert atom.lr._shap._shap_values.empty


def test_cross_validate():
    """Assert that the cross_validate method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert isinstance(atom.lr.cross_validate(), pd.DataFrame)
    assert isinstance(atom.lr.cross_validate(scoring="AP"), pd.DataFrame)


def test_dashboard_dataset_no_holdout():
    """Assert that an error is raised when there's no holdout set."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    with pytest.raises(ValueError, match=r".*No holdout data set.*"):
        atom.tree.dashboard(dataset="holdout")


def test_dashboard_invalid_dataset():
    """Assert that an error is raised when dataset is invalid."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    with pytest.raises(ValueError, match=r".*dataset parameter.*"):
        atom.tree.dashboard(dataset="invalid")


@patch("explainerdashboard.ExplainerDashboard")
@pytest.mark.parametrize("dataset", ["train", "both", "holdout"])
def test_dashboard_classification(func, dataset):
    """Assert that the dashboard method calls the underlying package."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("Tree")
    atom.tree.dashboard(dataset=dataset, filename="dashboard")
    func.assert_called_once()


@patch("explainerdashboard.ExplainerDashboard")
def test_dashboard_regression(func):
    """Assert that the dashboard method calls the underlying package."""
    atom = ATOMRegressor(X_reg, y_reg, holdout_size=0.1, random_state=1)
    atom.run("Tree")
    atom.tree.dashboard()
    func.assert_called_once()


def test_delete():
    """Assert that models can be deleted."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    atom.tree.delete()
    assert not atom.models
    assert not atom.metric


def test_evaluate_invalid_threshold():
    """Assert that an error is raised when the threshold is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    with pytest.raises(ValueError, match=r".*Value should lie.*"):
        atom.mnb.evaluate(threshold=0)


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


def test_evaluate_custom_metric():
    """Assert that custom metrics are accepted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    assert isinstance(atom.mnb.evaluate("roc_auc_ovo"), pd.Series)


def test_evaluate_threshold():
    """Assert that the threshold parameter changes the predictions."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("RF")
    pred_1 = atom.rf.evaluate(threshold=0.01)
    pred_2 = atom.rf.evaluate(threshold=0.99)
    assert not pred_1.equals(pred_2)


def test_evaluate_sample_weight():
    """Assert that the sample_weight parameter changes the predictions."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("RF")
    pred_1 = atom.rf.evaluate(sample_weight=None)
    pred_2 = atom.rf.evaluate(sample_weight=list(range(len(atom.y_test))))
    assert not pred_1.equals(pred_2)


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


def test_full_train_no_holdout():
    """Assert that an error is raised when include_holdout=True with no set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB")
    with pytest.raises(ValueError, match=r".*no holdout data set.*"):
        atom.lgb.full_train(include_holdout=True)


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


def test_full_train_clear():
    """Assert that the clear method is called."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    print(atom.tree.predict_log_proba_test)
    assert atom.tree._pred[7] is not None
    atom.tree.full_train()
    assert atom.tree._pred[7] is None


def test_full_train_new_mlflow_run():
    """Assert that a new mlflow run is created."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    run = atom.gnb._run
    atom.gnb.full_train()
    assert atom.gnb._run is not run


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
    atom.mnb.save_estimator("auto")
    assert glob.glob("MultinomialNB")


def test_transform():
    """Assert that new data can be transformed by the model's pipeline."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    atom.run("LR")
    X = atom.lr.transform(X10_str)
    assert len(X.columns) > 3  # Data is one-hot encoded
    assert check_scaling(X)  # Data is scaled
