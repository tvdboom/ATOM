# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for basemodel.py

"""

import glob
from unittest.mock import patch

import pandas as pd
import pytest
import requests
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.pruners import PatientPruner
from optuna.samplers import NSGAIISampler
from optuna.study import Study
from ray import serve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, recall_score
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier

from atom import ATOMClassifier, ATOMModel, ATOMRegressor
from atom.utils.utils import check_is_fitted, check_scaling, rnd

from .conftest import (
    X10_str, X_bin, X_class, X_idx, X_label, X_reg, y10, y10_str, y_bin,
    y_class, y_idx, y_label, y_multiclass, y_reg,
)


# Test magic methods ================================== >>

def test_scaler():
    """Assert that a scaler is made for models that need scaling."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LGB", "LDA"], est_params={"LGB": {"n_estimators": 5}})
    assert atom.lgb.scaler and not atom.lda.scaler


def test_repr():
    """Assert that the __repr__ method works as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LDA")
    assert str(atom.lda).startswith("LinearDiscriminantAnalysis")


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
    pd.testing.assert_series_equal(atom.tree["alcohol"], atom["alcohol"])
    pd.testing.assert_series_equal(atom.tree[0], atom[0])
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
            }
        },
    )


def test_est_params_removed_from_ht():
    """Assert that params in est_params are dropped from the optimization."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB", n_trials=1, est_params={"n_estimators": 5})
    assert "n_estimators" not in atom.lgb.trials.params[0]


def test_hyperparameter_tuning_with_no_hyperparameters():
    """Assert that the optimization is skipped when there are no hyperparameters."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models="BNB", n_trials=10, est_params={"alpha": 1.0, "fit_prior": True})
    assert atom.bnb.trials is None


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
    assert atom.xgb.trials["score"].sum() > 0  # All scores are positive


@patch("optuna.study.Study.get_trials")
def test_empty_study(func):
    """Assert that the optimization is skipped when there are no completed trials."""
    func.return_value = []  # No successful trials

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models="tree", n_trials=1)
    assert atom.tree.best_trial is None


def test_ht_with_pipeline():
    """Assert that the hyperparameter tuning works with a transformer pipeline."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)
    atom.encode()
    atom.run("lr", n_trials=1)
    assert atom.lr.trials is not None


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
        n_trials=10,
        ht_params={
            "distributions": {"max_iter": IntDistribution(5, 15)},
            "pruner": PatientPruner(None, patience=1),
        },
    )
    assert "PRUNED" in atom.sgd.trials["state"].values


def test_sample_weight_fit():
    """Assert that sample weights can be used with the BO."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="LGB",
        n_trials=1,
        est_params={
            "n_estimators": 5,
            "sample_weight_fit": list(range(len(atom.y_train))),
        },
    )


def test_custom_cv():
    """Assert that trials with a custom cv work for both tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("dummy", n_trials=1, ht_params={"cv": KFold(n_splits=3)})


def test_skip_duplicate_calls():
    """Assert that trials with the same parameters skip the calculation."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("dummy", n_trials=5)
    assert atom.dummy.trials["score"].nunique() < len(atom.dummy.trials["score"])


def test_trials_stored_correctly():
    """Assert that the trials attribute has same params as trial object."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", n_trials=3)
    assert atom.lr.trials.loc[2]["params"] == atom.lr.study.trials[2].params


@patch("mlflow.log_params")
def test_nested_runs_to_mlflow(mlflow):
    """Assert that the trials are logged to mlflow as nested runs."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.log_ht = True
    atom.run("Tree", n_trials=3)
    assert mlflow.call_count == 4  # n_trials + fit


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
    atom.log_model = True
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
    assert atom.tree.trials is None
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
    assert atom.lgb.bootstrap is None
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


def test_study_property():
    """Assert that the study property returns optuna's study."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=0)
    assert atom.tree.study is None
    atom.run("Tree", n_trials=1)
    assert isinstance(atom.tree.study, Study)


def test_trials_property():
    """Assert that the trials property returns an overview of the trials."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=0)
    assert atom.tree.trials is None
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


def test_score_ht_property():
    """Assert that the score_ht property returns the study's best score."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=0)
    assert atom.tree.score_ht is None
    atom.run("Tree", n_trials=5)
    assert isinstance(atom.tree.score_ht, float)


def test_time_ht_property():
    """Assert that the time_ht property returns the study's time."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=0)
    assert atom.tree.time_ht is None
    atom.run("Tree", n_trials=5)
    assert isinstance(atom.tree.time_ht, float)


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


def test_score_train_test_holdout_properties():
    """Assert that the score_train, score_test and score_holdout properties work."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.2, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.score_train, float)
    assert isinstance(atom.tree.score_test, float)
    assert isinstance(atom.tree.score_holdout, float)


def test_time_fit_property():
    """Assert that the time_fit property works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.time_fit, float)


def test_bootstrap_property():
    """Assert that the bootstrap property returns the bootstrap results."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom.tree.bootstrap is None
    atom.run("Tree", n_bootstrap=3)
    assert len(atom.tree.bootstrap) == 3


def test_score_bootstrap_property():
    """Assert that the score_bootstrap property works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom.tree.score_bootstrap is None
    atom.run("Tree", n_bootstrap=3)
    assert isinstance(atom.tree.score_bootstrap, float)


def test_time_bootstrap_property():
    """Assert that the time_bootstrap property works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom.tree.time_bootstrap is None
    atom.run("Tree", n_bootstrap=3)
    assert isinstance(atom.tree.time_bootstrap, float)


def test_time_property():
    """Assert that the time property returns the total time."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert isinstance(atom.tree.time, float)


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
    pd.testing.assert_frame_equal(atom.dataset, atom.mnb.dataset)
    assert check_scaling(atom.lr.dataset)


def test_train_property():
    """Assert that the train property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    pd.testing.assert_frame_equal(atom.train, atom.mnb.train)
    assert check_scaling(atom.lr.train)


def test_test_property():
    """Assert that the test property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    pd.testing.assert_frame_equal(atom.test, atom.mnb.test)
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
    pd.testing.assert_frame_equal(atom.X, atom.mnb.X)
    assert check_scaling(atom.lr.X)


def test_y_property():
    """Assert that the y property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    pd.testing.assert_series_equal(atom.y, atom.mnb.y)
    pd.testing.assert_series_equal(atom.y, atom.lr.y)


def test_X_train_property():
    """Assert that the X_train property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    pd.testing.assert_frame_equal(atom.X_train, atom.mnb.X_train)
    assert check_scaling(atom.lr.X_train)


def test_X_test_property():
    """Assert that the X_test property returns scaled data if needed."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    pd.testing.assert_frame_equal(atom.X_test, atom.mnb.X_test)
    assert check_scaling(atom.lr.X_test)


def test_X_holdout_property():
    """Assert that the X_holdout property is calculated."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    pd.testing.assert_frame_equal(atom.mnb.X_holdout, atom.mnb.holdout.iloc[:, :-1])


def test_y_train_property():
    """Assert that the y_train property is returned unchanged."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["MNB", "LR"])
    pd.testing.assert_series_equal(atom.y_train, atom.mnb.y_train)
    pd.testing.assert_series_equal(atom.y_train, atom.lr.y_train)


def test_y_holdout_property():
    """Assert that the y_holdout property is calculated."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("MNB")
    pd.testing.assert_series_equal(atom.mnb.y_holdout, atom.mnb.holdout.iloc[:, -1])


# Test prediction properties ======================================= >>

@pytest.mark.parametrize("dataset", ["train", "test", "holdout"])
def test_prediction_attributes_binary(dataset):
    """Assert that all prediction properties work for binary tasks."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("LR")
    assert isinstance(getattr(atom.lr, f"predict_{dataset}"), pd.Series)
    assert isinstance(getattr(atom.lr, f"predict_proba_{dataset}"), pd.DataFrame)
    assert isinstance(getattr(atom.lr, f"predict_log_proba_{dataset}"), pd.DataFrame)
    assert isinstance(getattr(atom.lr, f"decision_function_{dataset}"), pd.Series)
    assert isinstance(getattr(atom.lr, f"score_{dataset}"), float)


@pytest.mark.parametrize("dataset", ["train", "test", "holdout"])
def test_prediction_attributes_multilabel(dataset):
    """Assert that the prediction attributes change for multilabel tasks."""
    atom = ATOMClassifier(X_label, y=y_label, holdout_size=0.1, stratify=False)
    atom.run(["LR", "RF"])
    predict = getattr(atom.lr, f"predict_{dataset}")
    decision_function = getattr(atom.lr, f"decision_function_{dataset}")
    predict_proba = getattr(atom.rf, f"predict_proba_{dataset}")
    predict_log_proba = getattr(atom.rf, f"predict_log_proba_{dataset}")
    assert predict.shape[1] == y_label.shape[1]
    assert decision_function.shape[1] == y_label.shape[1]
    assert isinstance(predict_proba.index, pd.Index)
    assert isinstance(predict_log_proba.index, pd.Index)


@pytest.mark.parametrize("dataset", ["train", "test", "holdout"])
def test_prediction_attributes_multioutput(dataset):
    """Assert that the prediction attributes change for multioutput tasks."""
    atom = ATOMClassifier(X_class, y=y_multiclass, holdout_size=0.1, stratify=False)
    atom.run("RF")
    predict = getattr(atom.rf, f"predict_{dataset}")
    predict_proba = getattr(atom.rf, f"predict_proba_{dataset}")
    predict_log_proba = getattr(atom.rf, f"predict_log_proba_{dataset}")
    assert predict.shape[1] == y_multiclass.shape[1]
    assert isinstance(predict_proba.index, pd.MultiIndex)
    assert isinstance(predict_log_proba.index, pd.MultiIndex)


# Test prediction methods ========================================== >>

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


def test_score_regression():
    """Assert that the score returns r2 for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    r2 = r2_score(y_reg, atom.tree.predict(X_reg))
    assert atom.tree.score(X_reg, y_reg) == r2


def test_score_metric_is_None():
    """Assert that the score returns accuracy for classification tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert rnd(atom.tree.score(atom.X_test, atom.y_test)) == atom.tree.score_test


def test_score_custom_metric():
    """Assert that the score method works when sample weights are provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    recall = recall_score(y_bin, atom.tree.predict(X_bin))
    assert atom.tree.score(X_bin, y_bin, metric="recall") == recall


def test_score_with_sample_weight():
    """Assert that the score method works when sample weights are provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    score = atom.tree.score(X_bin, y_bin, sample_weight=list(range(len(y_bin))))
    assert isinstance(score, float)


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


def test_calibrate_clear():
    """Assert that the clear method is called."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    print(atom.tree.predict_log_proba_test)
    atom.tree.calibrate()
    assert "predict_log_proba_test" not in atom.tree.__dict__


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
    atom.plot_shap_beeswarm(display=False)
    assert "predict_test" in atom.lr.__dict__
    assert not atom.lr._shap._shap_values.empty
    atom.clear()
    assert "predict_test" not in atom.lr.__dict__
    assert atom.lr._shap._shap_values.empty


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


def test_create_dashboard_dataset_no_holdout():
    """Assert that an error is raised when there's no holdout set."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    with pytest.raises(ValueError, match=".*No holdout data set.*"):
        atom.tree.create_dashboard(dataset="holdout")


def test_create_dashboard_invalid_dataset():
    """Assert that an error is raised when dataset is invalid."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    with pytest.raises(ValueError, match=".*dataset parameter.*"):
        atom.tree.create_dashboard(dataset="invalid")


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


def test_evaluate_invalid_threshold_length():
    """Assert that an error is raised when the threshold is invalid."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("MNB")
    with pytest.raises(ValueError, match=".*should be equal to the number of target.*"):
        atom.mnb.evaluate(threshold=[0.5, 0.6])


def test_evaluate_invalid_threshold():
    """Assert that an error is raised when the threshold is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    with pytest.raises(ValueError, match=".*Value should lie.*"):
        atom.mnb.evaluate(threshold=0.0)


def test_evaluate_invalid_dataset():
    """Assert that an error is raised when the dataset is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    with pytest.raises(ValueError, match=".*Unknown value for the dataset.*"):
        atom.mnb.evaluate(dataset="invalid")


def test_evaluate_no_holdout():
    """Assert that an error is raised when there's no holdout set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    with pytest.raises(ValueError, match=".*No holdout data set.*"):
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

    atom = ATOMClassifier(
        X_label,
        y=y_label,
        holdout_size=0.1,
        stratify=False,
        random_state=1,
    )
    atom.run("MNB")
    scores = atom.mnb.evaluate(dataset=dataset)
    assert len(scores) == 7

    atom = ATOMRegressor(X_reg, y_reg, holdout_size=0.1, random_state=1)
    atom.run("OLS")
    scores = atom.ols.evaluate(dataset=dataset)
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
    atom.tree.full_train()
    assert "predict_log_proba_test" not in atom.tree.__dict__


def test_full_train_new_mlflow_run():
    """Assert that a new mlflow run is created."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run("GNB")
    run = atom.gnb._run
    atom.gnb.full_train()
    assert atom.gnb._run is not run


def test_get_best_threshold_no_predict_proba():
    """Assert that an error is raised when the model has no predict_proba."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("SVM")
    with pytest.raises(ValueError, match=".*with a predict_proba method.*"):
        atom.svm.get_best_threshold()


def test_get_best_threshold_invalid_dataset():
    """Assert that an error is raised when dataset is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Dummy")
    with pytest.raises(ValueError, match=".*dataset parameter.*"):
        atom.dummy.get_best_threshold("invalid")


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
    atom.clean()
    atom.run("LR")
    pd.testing.assert_frame_equal(atom.lr.inverse_transform(atom.lr.X), X_bin)


def test_save_estimator():
    """Assert that the save_estimator saves a pickle file."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.save_estimator("auto")
    assert glob.glob("MultinomialNB")


def test_serve():
    """Assert that the serve methods deploys a reachable endpoint."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("MNB")
    atom.mnb.serve()
    assert "200" in str(requests.get("http://127.0.0.1:8000/", json=X_bin.to_json()))
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
    assert all(-3 <= v <= 3 for v in X.values.ravel())  # Data is scaled
