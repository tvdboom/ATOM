"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for basemodel.py

"""

import glob
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
import requests
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.pruners import PatientPruner
from optuna.samplers import NSGAIISampler
from optuna.study import Study
from pandas.testing import assert_frame_equal, assert_series_equal
from ray import serve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, mean_absolute_error, mean_absolute_percentage_error, r2_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sktime.forecasting.base import ForecastingHorizon
from sktime.proba.normal import Normal

from atom import ATOMClassifier, ATOMForecaster, ATOMModel, ATOMRegressor
from atom.utils.utils import check_is_fitted, check_scaling

from .conftest import (
    X10_str, X_bin, X_class, X_ex, X_idx, X_label, X_reg, y10, y10_str, y_bin,
    y_class, y_ex, y_fc, y_idx, y_label, y_multiclass, y_reg,
)


# Test magic methods ================================== >>

def test_scaler():
    """Assert that a scaler is made for models that need scaling."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LGB", "LDA"], est_params={"LGB": {"n_estimators": 5}})
    assert atom.lgb.scaler
    assert not atom.lda.scaler


def test_repr():
    """Assert that the __repr__ method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LDA")
    assert str(atom.lda).startswith("LinearDiscriminantAnalysis")


def test_dir():
    """Assert that __dir__ contains all the extra attributes."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("dummy")
    assert all(attr in dir(atom.dummy) for attr in ("y", "age", "head"))


def test_getattr():
    """Assert that branch attributes can be called from a model."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.balance(strategy="smote")
    atom.run("Tree")
    assert isinstance(atom.tree.shape, tuple)
    assert isinstance(atom.tree.alcohol, pd.Series)
    assert isinstance(atom.tree.head(), pd.DataFrame)
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
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
    assert_series_equal(atom.tree["alcohol"], atom["alcohol"])
    assert_series_equal(atom.tree[0], atom[0])
    assert isinstance(atom.tree[["alcohol", "ash"]], pd.DataFrame)


# Test training ==================================================== >>

def test_est_params_invalid_param():
    """Assert that invalid parameters in est_params are caught."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models=["LR", "LGB"],
        n_trials=1,
        est_params={"test": 220, "LGB": {"n_estimators": 5}},
    )
    assert atom.models == "LGB"  # LGB passes since it accepts kwargs


def test_est_params_unknown_param_fit():
    """Assert that unknown parameters in est_params_fit are caught."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(RuntimeError, match=".*All models failed.*"):
        atom.run(["LR", "LGB"], n_trials=1, est_params={"test_fit": 220})


def test_custom_distributions_by_name():
    """Assert that the parameters to tune can be set by name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", n_trials=1, ht_params={"distributions": "max_iter"})
    assert list(atom.lr.best_params) == ["max_iter"]


def test_custom_distributions_by_name_excluded():
    """Assert that the parameters to tune can be excluded by name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("CNB", n_trials=1, ht_params={"distributions": "!fit_prior"})
    assert list(atom.cnb.best_params) == ["alpha", "norm"]


def test_custom_distributions_name_is_invalid():
    """Assert that an error is raised when an invalid parameter is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*is not a predefined hyperparameter.*"):
        atom.run(
            models="LR",
            n_trials=1,
            ht_params={"distributions": "invalid"},
            errors="raise",
        )


def test_custom_distributions_is_dist():
    """Assert that the custom distributions are for all models."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LR",
        n_trials=1,
        ht_params={"distributions": {"max_iter": IntDistribution(10, 20)}},
    )
    assert list(atom.lr.best_params) == ["max_iter"]


def test_custom_distributions_include_and_excluded():
    """Assert that an error is raised when parameters are included and excluded."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*either include or exclude.*"):
        atom.run(
            models="LR",
            n_trials=1,
            ht_params={"distributions": ["!max_iter", "penalty"]},
            errors="raise",
        )


def test_custom_distributions_meta_estimators():
    """Assert that meta-estimators can be tuned normally."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run(
        models=ATOMModel(
            estimator=ClassifierChain(LogisticRegression(), cv=2),
            native_multilabel=True,
        ),
        n_trials=1,
        ht_params={
            "distributions": {
                "order": CategoricalDistribution([(0, 1, 2, 3), (1, 0, 3, 2)]),
                "base_estimator__solver": CategoricalDistribution(["lbfgs", "newton-cg"]),
            },
        },
    )


def test_est_params_removed_from_ht():
    """Assert that params in est_params are dropped from the optimization."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", n_trials=1, est_params={"n_estimators": 5})
    assert "n_estimators" not in atom.lgb.trials


def test_hyperparameter_tuning_with_no_hyperparameters():
    """Assert that the optimization is skipped when there are no hyperparameters."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models="BNB", n_trials=10, est_params={"alpha": 1.0, "fit_prior": True})
    assert not hasattr(atom.bnb, "trials")


def test_multi_objective_optimization():
    """Assert that hyperparameter tuning works for multi-metric runs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR", metric=["f1", "auc"], n_trials=1)
    assert atom.lr.study.sampler.__class__ == NSGAIISampler


def test_hyperparameter_tuning_with_plot():
    """Assert that you can plot the hyperparameter tuning as it runs."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models=["LDA", "lSVM", "SVM"], n_trials=10, ht_params={"plot": True})


def test_xgb_optimizes_score():
    """Assert that the XGB model optimizes the score."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="XGB",
        n_trials=10,
        est_params={"n_estimators": 10},
        ht_params={"pruner": PatientPruner(None, patience=1)},
    )
    assert atom.xgb.trials["f1"].sum() > 0  # All scores are positive


@patch("optuna.study.Study.get_trials")
def test_empty_study(func):
    """Assert that the optimization is skipped when there are no completed trials."""
    func.return_value = []  # No successful trials

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models="tree", n_trials=1)
    assert not hasattr(atom.tree, "study")


def test_ht_with_pipeline():
    """Assert that the hyperparameter tuning works with a transformer pipeline."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    atom.run("lr", n_trials=1)
    assert hasattr(atom.lr, "trials")


def test_ht_with_multilabel():
    """Assert that the hyperparameter tuning works with multilabel tasks."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("SGD", n_trials=1, est_params={"max_iter": 5})
    atom.multioutput = None
    atom.run("MLP", n_trials=1, est_params={"max_iter": 5})


def test_ht_with_multioutput():
    """Assert that the hyperparameter tuning works with multioutput tasks."""
    atom = ATOMClassifier(X_class, y=y_multiclass, stratify=False, random_state=1)
    atom.run("SGD", n_trials=1, est_params={"max_iter": 5})


def test_ht_with_pruning():
    """Assert that trials can be pruned."""
    atom = ATOMClassifier(X_bin, y=y_bin, random_state=1)
    atom.run(
        models="SGD",
        n_trials=7,
        ht_params={
            "distributions": {"max_iter": IntDistribution(5, 15)},
            "pruner": PatientPruner(None, patience=1),
        },
    )
    assert "PRUNED" in atom.sgd.trials["state"].unique()


def test_custom_cv():
    """Assert that trials with a custom cv work for both tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("dummy", n_trials=1, ht_params={"cv": KFold(n_splits=3)})


def test_skip_duplicate_calls():
    """Assert that trials with the same parameters skip the calculation."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("dummy", n_trials=5)
    assert atom.dummy.trials["f1"].nunique() < len(atom.dummy.trials["f1"])


def test_trials_stored_correctly():
    """Assert that the `trials` attribute has the same params as the trial object."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", n_trials=3, ht_params={"distributions": ["penalty", "C"]})
    assert atom.lr.trials.loc[2, "penalty"] == atom.lr.study.trials[2].params["penalty"]
    assert atom.lr.trials.loc[2, "C"] == atom.lr.study.trials[2].params["C"]


@patch("mlflow.log_params")
def test_nested_runs_to_mlflow(mlflow):
    """Assert that the trials are logged to mlflow as nested runs."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.log_ht = True
    atom.run("Tree", n_trials=1)
    assert mlflow.call_count == 2  # n_trials + fit


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
    atom.run("LGB")
    assert mlflow.call_count > 10


@patch("mlflow.sklearn.log_model")
def test_run_log_models_to_mlflow(mlflow):
    """Assert that models are logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("LGB")
    assert mlflow.call_count == 1


@patch("mlflow.log_input")
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


def test_continued_hyperparameter_tuning():
    """Assert that the hyperparameter_tuning method can be recalled."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert not hasattr(atom.tree, "trials")
    atom.tree.hyperparameter_tuning(3)
    assert len(atom.tree.trials) == 3
    atom.tree.hyperparameter_tuning(3)
    assert len(atom.tree.trials) == 6
    atom.tree.hyperparameter_tuning(2, reset=True)
    assert len(atom.tree.trials) == 2


def test_continued_bootstrapping():
    """Assert that the bootstrapping method can be recalled."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", est_params={"n_estimators": 5})
    assert not hasattr(atom.lgb, "bootstrap")
    atom.lgb.bootstrapping(3)
    assert len(atom.lgb.bootstrap) == 3
    atom.lgb.bootstrapping(3)
    assert len(atom.lgb.bootstrap) == 6
    atom.lgb.bootstrapping(3, reset=True)
    assert len(atom.lgb.bootstrap) == 3


# Test utility properties ========================================== >>

def test_name_property():
    """Assert that the name property can be set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree_2")
    assert atom.tree_2.name == "Tree_2"
    atom.tree_2.name = ""
    assert atom.tree.name == "Tree"
    atom.tree.name = "Tree_3"
    assert atom.tree_3.name == "Tree_3"
    atom.tree_3.name = "4"
    assert atom.tree_4.name == "Tree_4"


@patch("mlflow.MlflowClient.set_tag")
def test_name_property_to_mlflow(mlflow):
    """Assert that the new name is stored in mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("Tree")
    atom.tree.name = "2"
    mlflow.assert_called_with(atom.tree_2._run.info.run_id, "mlflow.runName", "Tree_2")


def test_og_property():
    """Assert that the og property returns the original Branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom.tree.og is atom.og


def test_branch_property():
    """Assert that the branch property returns the Branch."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom.tree.branch is atom.branch


def test_run_property():
    """Assert that the run property returns the mlflow run."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("Tree")
    assert hasattr(atom.tree, "run")


def test_study_property():
    """Assert that the study property returns optuna's study."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=0)
    assert not hasattr(atom.tree, "study")
    atom.run("Tree", n_trials=1)
    assert isinstance(atom.tree.study, Study)


def test_trials_property():
    """Assert that the trials property returns an overview of the trials."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=0)
    assert not hasattr(atom.tree, "trials")
    atom.run("Tree", n_trials=1)
    assert isinstance(atom.tree.trials, pd.DataFrame)


def test_best_trial_property():
    """Assert that the best_trial property can be set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=5)
    assert atom.tree.best_trial.number == 1
    atom.tree.best_trial = 4
    assert atom.tree.best_trial.number == 4


def test_best_trial_property_invalid():
    """Assert that an error is raised when best_trial is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=5)
    with pytest.raises(ValueError, match=".*should be a trial number.*"):
        atom.tree.best_trial = 22


def test_best_params_property():
    """Assert that the best_params property returns the hyperparameters."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=0)
    assert atom.tree.best_params == {}
    atom.run("Tree", n_trials=5)
    assert isinstance(atom.tree.best_params, dict)


def test_estimator_property():
    """Assert that the estimator property returns the estimator."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.estimator, DecisionTreeClassifier)
    assert check_is_fitted(atom.tree.estimator)


def test_evals_property():
    """Assert that the estimator property returns the estimator."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", est_params={"n_estimators": 5})
    assert len(atom.lgb.evals) == 2


def test_bootstrap_property():
    """Assert that the bootstrap property returns the bootstrap results."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert not hasattr(atom.tree, "bootstrap")
    atom.run("Tree", n_bootstrap=3)
    assert len(atom.tree.bootstrap) == 3


def test_feature_importance_property():
    """Assert that the feature_importance property works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert len(atom.tree.feature_importance) == X_bin.shape[1]

    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("LDA")
    assert len(atom.lda.feature_importance) == X_label.shape[1]

    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    atom.run("LDA")
    assert len(atom.lda.feature_importance) == X_class.shape[1]


def test_results_property():
    """Assert that the property returns an overview of the model's results."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.results, pd.Series)


# Test data properties ============================================= >>

def test_pipeline_property():
    """Assert that the pipeline property returns the scaler as well."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.run(["LR", "Tree"])
    assert len(atom.lr.pipeline) == 2
    assert len(atom.tree.pipeline) == 1


def test_dataset_property():
    """Assert that the dataset property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert_frame_equal(atom.dataset, atom.mnb.dataset)
    assert check_scaling(atom.lr.dataset)


def test_train_property():
    """Assert that the train property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert_frame_equal(atom.train, atom.mnb.train)
    assert check_scaling(atom.lr.train)


def test_test_property():
    """Assert that the test property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert_frame_equal(atom.test, atom.mnb.test)
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
    assert_frame_equal(atom.X, atom.mnb.X)
    assert check_scaling(atom.lr.X)


def test_y_property():
    """Assert that the y property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert_series_equal(atom.y, atom.mnb.y)
    assert_series_equal(atom.y, atom.lr.y)


def test_X_train_property():
    """Assert that the X_train property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert_frame_equal(atom.X_train, atom.mnb.X_train)
    assert check_scaling(atom.lr.X_train)


def test_X_test_property():
    """Assert that the X_test property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert_frame_equal(atom.X_test, atom.mnb.X_test)
    assert check_scaling(atom.lr.X_test)


def test_X_holdout_property():
    """Assert that the X_holdout property is calculated."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    assert_frame_equal(atom.mnb.X_holdout, atom.mnb.holdout.iloc[:, :-1])


def test_y_train_property():
    """Assert that the y_train property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    assert_series_equal(atom.y_train, atom.mnb.y_train)
    assert_series_equal(atom.y_train, atom.lr.y_train)


def test_y_holdout_property():
    """Assert that the y_holdout property is calculated."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    assert_series_equal(atom.mnb.y_holdout, atom.mnb.holdout.iloc[:, -1])


def test_shape_property():
    """Assert that the shape property returns the shape of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    atom.run("MNB")
    assert atom.mnb.shape == (len(X_bin), X_bin.shape[1] - 1)


def test_columns_property():
    """Assert that the columns property returns the columns of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    atom.run("MNB")
    assert len(atom.mnb.columns) == len(atom.columns) - 2


def test_n_columns_property():
    """Assert that the n_columns property returns the number of columns."""
    atom = ATOMClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    atom.run("MNB")
    assert atom.mnb.n_columns == atom.n_columns - 2


def test_features_property():
    """Assert that the features property returns the features of the dataset."""
    atom = ATOMClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    atom.run("MNB")
    assert len(atom.mnb.features) == len(atom.features) - 2


def test_n_features_property():
    """Assert that the n_features property returns the number of features."""
    atom = ATOMClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    atom.run("MNB")
    assert atom.mnb.n_features == atom.n_features - 2


def test_all_property():
    """Assert that the _all property returns the dataset + holdout."""
    atom = ATOMRegressor(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("OLS")
    assert len(atom.ols.dataset) != len(X_bin)
    assert len(atom.ols._all) == len(X_bin)


# Test utility methods ============================================= >>

def test_calibrate_invalid_task():
    """Assert than an error is raised when task="regression"."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("OLS")
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        atom.ols.calibrate()


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


def test_calibrate_new_mlflow_run():
    """Assert that a new mlflow run is created."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    run = atom.gnb._run
    atom.gnb.calibrate()
    assert atom.gnb._run is not run


def test_clear():
    """Assert that the clear method resets the model's attributes."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("SGD", est_params={"max_iter": 5})
    atom.plot_shap_beeswarm(display=False)
    atom.evaluate(rows="holdout")
    assert atom.sgd._evals
    assert not atom.sgd._shap._shap_values.empty
    assert "holdout" in atom.sgd.branch.__dict__
    atom.clear()
    assert not atom.sgd._evals
    assert atom.sgd._shap._shap_values.empty
    assert "holdout" not in atom.sgd.branch.__dict__


@patch("gradio.Interface")
def test_create_app(interface):
    """Assert that the create_app method calls the underlying package."""
    atom = ATOMClassifier(X10_str, y10_str, random_state=1)
    atom.clean()
    atom.encode()
    atom.run("Tree")
    atom.tree.create_app()
    interface.assert_called_once()


def test_create_dashboard_multioutput():
    """Assert that the method is unavailable for multioutput tasks."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    atom.run("Tree")
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        atom.tree.create_dashboard()


@patch("explainerdashboard.ExplainerDashboard")
def test_create_dashboard_binary(func):
    """Assert that the create_dashboard method calls the underlying package."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("LR")
    atom.lr.create_dashboard(dataset="holdout", filename="dashboard")
    func.assert_called_once()


@patch("explainerdashboard.ExplainerDashboard")
def test_create_dashboard_multiclass(func):
    """Assert that the create_dashboard method calls the underlying package."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("Tree")
    atom.tree.create_dashboard()
    func.assert_called_once()


@patch("explainerdashboard.ExplainerDashboard")
def test_create_dashboard_regression(func):
    """Assert that the create_dashboard method calls the underlying package."""
    atom = ATOMRegressor(X_reg, y_reg, holdout_size=0.1, random_state=1)
    atom.run("Tree")
    atom.tree.create_dashboard(dataset="both")
    func.assert_called_once()


def test_cross_validate():
    """Assert that the cross_validate method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert isinstance(atom.lr.cross_validate(), pd.DataFrame)
    assert isinstance(atom.lr.cross_validate(scoring="AP"), pd.DataFrame)


def test_cross_validate_ts():
    """Assert that the cross_validate method works for forecast tasks."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run("NF")
    assert isinstance(atom.nf.cross_validate(), pd.DataFrame)
    assert isinstance(atom.nf.cross_validate(scoring="mae"), pd.DataFrame)


def test_evaluate_invalid_threshold_length():
    """Assert that an error is raised when the threshold is invalid."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("MNB")
    with pytest.raises(ValueError, match=".*should be equal to the number of target.*"):
        atom.mnb.evaluate(threshold=[0.5, 0.6])


def test_evaluate_metric_None():
    """Assert that the evaluate method works when metric is empty."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    scores = atom.mnb.evaluate()
    assert len(scores) == 9

    atom = ATOMClassifier(X_class, y_class, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    scores = atom.mnb.evaluate()
    assert len(scores) == 6

    atom = ATOMClassifier(
        X_label,
        y=y_label,
        holdout_size=0.1,
        stratify=False,
        random_state=1,
    )
    atom.run("MNB")
    scores = atom.mnb.evaluate()
    assert len(scores) == 7

    atom = ATOMRegressor(X_reg, y_reg, holdout_size=0.1, random_state=1)
    atom.run("OLS")
    scores = atom.ols.evaluate()
    assert len(scores) == 5


def test_evaluate_custom_metric():
    """Assert that custom metrics are accepted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    assert isinstance(atom.mnb.evaluate("roc_auc_ovo"), pd.Series)


def test_evaluate_threshold():
    """Assert that the threshold parameter changes the predictions."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("RF", est_params={"n_estimators": 5})
    pred_1 = atom.rf.evaluate(threshold=0.01)
    pred_2 = atom.rf.evaluate(threshold=0.99)
    assert not pred_1.equals(pred_2)


def test_evaluate_threshold_multilabel():
    """Assert that the threshold parameter accepts a list as threshold."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.evaluate(threshold=[0.4, 0.6, 0.8, 0.9]), pd.Series)


def test_export_pipeline():
    """Assert that the pipeline can be retrieved from the model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.clean()
    atom.run("LR")
    assert len(atom.lr.export_pipeline()) == 3


def test_full_train_no_holdout():
    """Assert that an error is raised when include_holdout=True with no set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB")
    with pytest.raises(ValueError, match=".*holdout data set.*"):
        atom.lgb.full_train(include_holdout=True)


def test_full_train():
    """Assert that the full_train method trains on the test set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB")
    atom.lgb.full_train()
    assert atom.lgb.score("test") == 1.0  # Perfect score on test


def test_full_train_holdout():
    """Assert that the full_train method trains on the holdout set."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.2, random_state=1)
    atom.run("Tree")
    atom.tree.full_train(include_holdout=True)
    assert atom.tree.score("holdout") == 1.0  # Perfect score on holdout


def test_full_train_new_mlflow_run():
    """Assert that a new mlflow run is created."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    run = atom.gnb._run
    atom.gnb.full_train()
    assert atom.gnb._run is not run


def test_get_best_threshold_binary():
    """Assert that the get_best_threshold method works for binary tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert 0 < atom.lr.get_best_threshold() < 1


def test_get_best_threshold_multilabel():
    """Assert that the get_best_threshold method works for multilabel tasks."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("LR")
    assert len(atom.lr.get_best_threshold()) == len(atom.target)


def test_inverse_transform():
    """Assert that the inverse_transform method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    atom.run("LR")
    assert_frame_equal(atom.lr.inverse_transform(atom.lr.X), X_bin)


def test_inverse_transform_output():
    """Assert that the output type is determined by the data engine."""
    atom = ATOMClassifier(X_bin, y_bin, engine="polars", random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.inverse_transform(X_bin), pl.DataFrame)


def test_save_estimator():
    """Assert that the save_estimator saves a pickle file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.save_estimator("auto")
    assert glob.glob("MultinomialNB.pkl")


def test_serve():
    """Assert that the serve method deploys a reachable endpoint."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.serve()
    response = requests.get("http://127.0.0.1:8000/", json=X_bin.to_json(), timeout=5)
    assert response.status_code == 200
    serve.shutdown()


def test_register_no_experiment():
    """Assert that an error is raised when there is no experiment."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    with pytest.raises(PermissionError, match=".*mlflow experiment.*"):
        atom.mnb.register()


@patch("mlflow.register_model")
@patch("mlflow.MlflowClient.transition_model_version_stage")
def test_register(mlflow, client):
    """Assert that the register saves the model to a stage."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("MNB")
    atom.mnb.register()
    mlflow.assert_called_once()
    client.assert_called_once()


def test_transform():
    """Assert that new data can be transformed by the model's pipeline."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    atom.run("LR")
    X = atom.lr.transform(X10_str)
    assert len(X.columns) > 3  # Data is one-hot encoded
    assert all(-3 <= v <= 3 for v in X.to_numpy().ravel())  # Data is scaled


def test_transform_output():
    """Assert that the output type is determined by the data engine."""
    atom = ATOMClassifier(X_bin, y_bin, engine="polars", random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.transform(X_bin), pl.DataFrame)


# Test ClassRegModel ================================================== >>

def test_classreg_get_tags():
    """Assert that the get_tags method returns the tags."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert isinstance(atom.lr.get_tags(), dict)


def test_predictions_from_index():
    """Assert that predictions can be made from data indices."""
    atom = ATOMClassifier(X_idx, y_idx, index=True, holdout_size=0.1, random_state=1)
    atom.run("LR")
    assert isinstance(atom.lr.decision_function(("index_4", "index_5")), pd.Series)
    assert isinstance(atom.lr.predict(["index_4", "index_8"]), pd.Series)
    assert isinstance(atom.lr.predict_log_proba(-100), pd.DataFrame)
    assert isinstance(atom.lr.predict_proba("index_4"), pd.DataFrame)
    assert isinstance(atom.lr.score(slice(10, 20)), float)


def test_transformations_first():
    """Assert that the transformations are applied before predicting."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode(max_onehot=None)
    atom.run("Tree")
    assert isinstance(atom.tree.predict(X10_str), pd.Series)


def test_data_is_scaled():
    """Assert that the data is scaled for models that need it."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert sum(atom.lr.predict(X_bin)) > 0  # Always 0 if not scaled


def test_predictions_from_new_data():
    """Assert that predictions can be made from new data."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    assert isinstance(atom.lr.predict(X_bin), pd.Series)
    assert isinstance(atom.lr.predict_proba(X_bin), pd.DataFrame)


def test_prediction_from_multioutput():
    """Assert that predictions can be made for multioutput datasets."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    atom.run("LR")
    assert isinstance(atom.lr.predict_proba(X_class).index, pd.MultiIndex)


def test_prediction_inverse_transform():
    """Assert that the predict method can return the inversely transformed data."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.scale(columns=-1)
    atom.run("Tree")
    assert check_scaling(atom.tree.predict(X_reg, inverse=False))
    assert not check_scaling(atom.tree.predict(X_reg, inverse=True))


def test_score_regression():
    """Assert that the score returns r2 for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, shuffle=False, random_state=1)
    atom.run("Tree")
    r2 = r2_score(y_reg, atom.tree.predict(X_reg))
    assert atom.tree.score(X_reg, y_reg) == r2


def test_score_metric_is_None():
    """Assert that the score method returns the default metric."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    atom.run("Tree")
    f1 = f1_score(y_bin, atom.tree.predict(X_bin))
    assert atom.tree.score(X_bin, y_bin) == f1


def test_score_custom_metric():
    """Assert that the score method works for a custom scorer."""
    atom = ATOMClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    atom.run("Tree")
    recall = recall_score(y_bin, atom.tree.predict(X_bin))
    assert atom.tree.score(X_bin, y_bin, metric="recall") == recall


def test_score_with_sample_weight():
    """Assert that the score method works when sample weights are provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    score = atom.tree.score(X_bin, y_bin, sample_weight=list(range(len(y_bin))))
    assert isinstance(score, float)


# Test ForecastModel =============================================== >>

def test_forecast_get_tags():
    """Assert that the get_tags method returns the tags."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run("NF")
    assert isinstance(atom.nf.get_tags(), dict)


def test_predictions_only_fh():
    """Assert that predictions can be made using only the fh."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run("NF")
    assert isinstance(atom.nf.predict(fh=range(10)), pd.Series)
    assert isinstance(atom.nf.predict_interval(fh=ForecastingHorizon([1, 2])), pd.DataFrame)


def test_predictions_with_exogenous():
    """Assert that predictions can be made with exogenous variables."""
    atom = ATOMForecaster(X_ex, y=y_ex, random_state=1)
    atom.run("NF")
    assert isinstance(atom.nf.predict_proba(fh=range(10), X=X_ex.iloc[:10]), Normal)
    assert isinstance(atom.nf.predict_quantiles(fh=range(10), X=X_ex.iloc[:10]), pd.DataFrame)
    assert isinstance(atom.nf.predict_var(fh=range(10), X=X_ex.iloc[:10]), pd.DataFrame)


def test_ts_prediction_inverse_transform():
    """Assert that the predict method can return the inversely transformed data."""
    atom = ATOMForecaster(X_ex, y=y_ex, random_state=1)
    atom.scale(columns=-1)
    atom.run("NF")
    assert check_scaling(atom.nf.predict(y_ex, inverse=False))
    assert not check_scaling(atom.nf.predict(y_ex, inverse=True))
    assert check_scaling(atom.nf.predict_interval(y_ex, inverse=False))
    assert not check_scaling(atom.nf.predict_interval(y_ex, inverse=True))


def test_predictions_with_y():
    """Assert that predictions can be made with y."""
    atom = ATOMForecaster(X_ex, y=y_ex, random_state=1)
    atom.run("NF")
    assert isinstance(atom.nf.predict_residuals(y=y_ex[:10], X=X_ex.iloc[:10]), pd.Series)


def test_score_ts_metric_is_None():
    """Assert that the score method returns the default metric."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run("NF")
    mape = mean_absolute_percentage_error(y_fc[:10], atom.nf.predict(range(10)))
    assert atom.nf.score(y_fc[:10]) == np.negative(mape)


def test_score_ts_custom_metric():
    """Assert that the score method works for a custom scorer."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run("NF")
    mae = mean_absolute_error(y_fc[:10], atom.nf.predict(range(10)))
    assert atom.nf.score(y_fc[:10], metric="mae") == np.negative(mae)
