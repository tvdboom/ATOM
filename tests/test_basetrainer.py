# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for basetrainer.py

"""

# Standard packages
import pytest
import pandas as pd
from sklearn.metrics import get_scorer, f1_score
from skopt.space.space import Integer, Categorical

# Own modules
from atom.training import TrainerClassifier, TrainerRegressor
from .utils import bin_train, bin_test, class_train, class_test, reg_train, reg_test


# Test _check_parameters ==================================================== >>

def test_model_is_string():
    """Assert that a string input is accepted."""
    trainer = TrainerClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == ["LR"]


def test_models_get_right_name():
    """Assert that the model names are transformed to the correct acronyms."""
    trainer = TrainerClassifier(["lR", "et", "CATB"], random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == ["LR", "ET", "CatB"]


def test_creation_model_subclasses():
    """Assert that the model subclasses are created correctly."""
    trainer = TrainerClassifier(["LR", "LDA"], random_state=1)
    trainer.run(bin_train, bin_test)
    assert hasattr(trainer, "LR") and hasattr(trainer, "lr")
    assert hasattr(trainer, "LDA") and hasattr(trainer, "lda")


def test_duplicate_models():
    """Assert that an error is raised when models contains duplicates."""
    trainer = TrainerClassifier(["lr", "LR", "lgb"], random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_only_task_models():
    """Assert that an error is raised for models at invalid task."""
    trainer = TrainerClassifier("OLS", random_state=1)  # Only regression
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)

    trainer = TrainerRegressor("LDA", random_state=1)  # Only classification
    pytest.raises(ValueError, trainer.run, reg_train, reg_test)


def test_n_calls_parameter_invalid():
    """Assert that an error is raised when n_calls is invalid."""
    trainer = TrainerClassifier("LR", n_calls=(2, 2), random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_n_calls_parameter_to_list():
    """Assert that n_calls is made a list."""
    trainer = TrainerClassifier(["LR", "LDA"], n_calls=7, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.n_calls == [7, 7]


def test_n_initial_points_parameter_invalid():
    """Assert that an error is raised when n_initial_points is invalid."""
    trainer = TrainerClassifier(
        models="LR", n_calls=2, n_initial_points=(2, 2), random_state=1
    )
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_n_initial_points_parameter_to_list():
    """Assert that n_initial_points is made a list."""
    trainer = TrainerClassifier(
        models=["LR", "LDA"], n_calls=2, n_initial_points=1, random_state=1
    )
    trainer.run(bin_train, bin_test)
    assert trainer.n_initial_points == [1, 1]


def test_bagging_parameter_invalid():
    """Assert that an error is raised when bagging is invalid."""
    trainer = TrainerClassifier("LR", bagging=(2, 2), random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_bagging_parameter_to_list():
    """Assert that bagging is made a list."""
    trainer = TrainerClassifier(["LR", "LDA"], bagging=2, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.bagging == [2, 2]


def test_dimensions_all_models():
    """Assert that the dimensions are passed to all models."""
    dim = [Integer(10, 100, name="n_estimators")]

    # If more than one model
    trainer = TrainerClassifier(
        models=["XGB", "LGB"], bo_params={"dimensions": dim}, random_state=1
    )
    trainer.run(bin_train, bin_test)
    assert trainer.bo_params["dimensions"] == {"XGB": dim, "LGB": dim}


def test_dimensions_per_model():
    """Assert that the dimensions are passed per model."""
    dim_1 = [Integer(100, 1000, name="max_iter")]
    dim_2 = [Integer(50, 100, name="n_estimators")]

    trainer = TrainerClassifier(
        models=["LR", "RF"],
        bo_params={"dimensions": {"lr": dim_1, "rf": dim_2}},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.bo_params["dimensions"] == {"LR": dim_1, "RF": dim_2}


def test_est_params_all_models():
    """Assert that est_params passes the parameters to all models."""
    trainer = TrainerClassifier(
        models=["XGB", "LGB"], est_params={"n_estimators": 220}, random_state=1
    )
    trainer.run(bin_train, bin_test)
    assert trainer.lgb.estimator.get_params()["n_estimators"] == 220
    assert trainer.xgb.estimator.get_params()["n_estimators"] == 220


def test_est_params_per_model():
    """Assert that est_params passes the parameters per model."""
    trainer = TrainerClassifier(
        models=["XGB", "LGB"],
        est_params={"xgb": {"n_estimators": 220}, "lgb": {"n_estimators": 200}},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.xgb.estimator.get_params()["n_estimators"] == 220
    assert trainer.lgb.estimator.get_params()["n_estimators"] == 200


def test_default_metric():
    """Assert that a default metric_ is assigned depending on the task."""
    trainer = TrainerClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1"

    trainer = TrainerClassifier("LR", random_state=1)
    trainer.run(class_train, class_test)
    assert trainer.metric == "f1_weighted"

    trainer = TrainerRegressor("LGB", random_state=1)
    trainer.run(reg_train, reg_test)
    assert trainer.metric == "r2"


# Test _prepare_metric ====================================================== >>

def test_metric_to_list():
    """Assert that the metric attribute is always a list."""
    trainer = TrainerClassifier("LR", metric="f1", random_state=1)
    trainer.run(bin_train, bin_test)
    assert isinstance(trainer.metric_, list)


def test_greater_is_better_parameter():
    """Assert that an error is raised if invalid length for greater_is_better."""
    trainer = TrainerClassifier(
        models="LR", metric="f1", greater_is_better=[True, False], random_state=1
    )
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_needs_proba_parameter():
    """Assert that an error is raised if invalid length for needs_proba."""
    trainer = TrainerClassifier(
        models="LR", metric="f1", needs_proba=[True, False], random_state=1
    )
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_needs_threshold_parameter():
    """Assert that an error is raised if invalid length for needs_threshold."""
    trainer = TrainerClassifier(
        models="LR", metric="f1", needs_threshold=[True, False], random_state=1
    )
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_metric_acronym():
    """"Assert that using the metric acronyms work."""
    trainer = TrainerClassifier("LR", metric="auc", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "roc_auc"


def test_invalid_scorer_name():
    """Assert that an error is raised when scorer name is invalid."""
    trainer = TrainerClassifier("LR", metric="test", random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_function_metric_parameter():
    """Assert that a function metric works."""
    trainer = TrainerClassifier("LR", metric=f1_score, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1_score"


def test_scorer_metric_parameter():
    """Assert that a scorer metric works."""
    trainer = TrainerClassifier("LR", metric=get_scorer("f1"), random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1"


# Test _params_to_attr ====================================================== >>

def test_data_already_set():
    """Assert that if there already is data, the call to run can be empty."""
    dataset = pd.concat([bin_train, bin_test]).reset_index(drop=True)

    trainer = TrainerClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    trainer.run()
    assert trainer.dataset.equals(dataset)
    assert trainer._idx == [len(bin_train), len(bin_test)]


def test_new_scaler():
    """Assert that the scaler is reset when new data is provided."""
    trainer = TrainerClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    scaler_1 = trainer.scaler

    # Rerun with new data
    trainer.run(bin_train.iloc[12:70, :], bin_test)
    scaler_2 = trainer.scaler

    # Compare the scalers transforming some data
    data = bin_test.iloc[:, :-1]
    assert not scaler_1.transform(data).equals(scaler_2.transform(data))


# Test _run ================================================================= >>

def test_sequence_parameters():
    """Assert that every model get his corresponding parameters."""
    trainer = TrainerClassifier(
        models=["LR", "LDA", "LGB"],
        n_calls=(2, 3, 4),
        n_initial_points=(1, 2, 3),
        bagging=[2, 5, 7],
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert len(trainer.LR.bo) == 2
    assert sum(trainer.LDA.bo.index.str.startswith("Initial")) == 2
    assert len(trainer.lgb.metric_bagging) == 7


def test_invalid_n_calls_parameter():
    """Assert that an error is raised for negative n_calls."""
    trainer = TrainerClassifier("LR", n_calls=-1, n_initial_points=1, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_scaler_is_created():
    """Assert the scaler is created for models that need scaling."""
    trainer = TrainerClassifier("LGB", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.scaler is not None


def test_custom_dimensions_for_bo():
    """Assert that the BO runs when custom dimensions are provided."""
    trainer = TrainerRegressor(
        models="ols",
        n_calls=5,
        bo_params={"dimensions": [Categorical([True, False], name="fit_intercept")]},
        random_state=1,
    )
    trainer.run(reg_train, reg_test)
    assert not trainer.ols.bo.empty


def test_error_handling():
    """Assert that models with errors are removed from pipeline."""
    trainer = TrainerClassifier(
        models=["LR", "LDA"], n_calls=4, n_initial_points=[2, -1], random_state=1
    )
    trainer.run(bin_train, bin_test)
    assert trainer.errors.get("LDA")
    assert "lDA" not in trainer.models
    assert "LDA" not in trainer.results.index


def test_all_models_failed():
    """Assert that an error is raised when all models failed."""
    trainer = TrainerClassifier(
        models="LR", n_calls=4, n_initial_points=-1, random_state=1
    )
    pytest.raises(RuntimeError, trainer.run, bin_train, bin_test)
