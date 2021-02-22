# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for basetrainer.py

"""

# Standard packages
import pytest
from sklearn.metrics import get_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from skopt.space.space import Integer, Categorical
from skopt.callbacks import TimerCallback

# Own modules
from atom.training import DirectClassifier, DirectRegressor
from atom.utils import PlotCallback
from .utils import bin_train, bin_test, class_train, class_test, reg_train, reg_test


# Test _check_parameters =========================================== >>

def test_model_is_predefined():
    """Assert that predefined models are accepted."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == "LR"


def test_model_is_custom():
    """Assert that custom models are accepted."""
    trainer = DirectClassifier(RandomForestClassifier, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == "RFC"


def test_models_get_right_name():
    """Assert that the model names are transformed to the correct acronyms."""
    trainer = DirectClassifier(["lR", "et", "CATB"], random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == ["LR", "ET", "CatB"]


def test_invalid_model_name():
    """Assert that an error is raised when the model is unknown."""
    trainer = DirectClassifier(models="invalid", random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_multiple_same_models():
    """Assert that the same model can used with different names."""
    trainer = DirectClassifier(["lr", "lr2", "lr_3"], random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.models == ["LR", "LR2", "LR_3"]


def test_only_task_models():
    """Assert that an error is raised for models at invalid task."""
    trainer = DirectClassifier("OLS", random_state=1)  # Only regression
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)

    trainer = DirectRegressor("LDA", random_state=1)  # Only classification
    pytest.raises(ValueError, trainer.run, reg_train, reg_test)


def test_duplicate_models():
    """Assert that duplicate inputs are ignored."""
    trainer = DirectClassifier(["lr", "LR", "lgb"], random_state=1)
    trainer.run(bin_train, bin_test)
    assert len(trainer.models) == 2


def test_default_metric():
    """Assert that a default metric is assigned depending on the task."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1"

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(class_train, class_test)
    assert trainer.metric == "f1_weighted"

    trainer = DirectRegressor("LGB", random_state=1)
    trainer.run(reg_train, reg_test)
    assert trainer.metric == "r2"


def test_default_mapping_assignment():
    """Assert that a default mapping is assigned."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.mapping == {"0": 0, "1": 1}


def test_sequence_parameters_invalid_length():
    """Assert that an error is raised when the length is invalid."""
    trainer = DirectClassifier("LR", n_calls=(2, 2), random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_sequence_parameters_under_zero():
    """Assert that an error is raised for values <0."""
    trainer = DirectClassifier("LR", n_calls=-2, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_n_initial_points_parameter_is_zero():
    """Assert that an error is raised when n_initial_points=0."""
    trainer = DirectClassifier("LR", n_calls=2, n_initial_points=0, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_base_estimator_default():
    """Assert that GP is the default base estimator."""
    trainer = DirectClassifier("LR", n_calls=5, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer._base_estimator == "GP"


def test_base_estimator_invalid():
    """Assert that an error is raised when the base estimator is invalid."""
    trainer = DirectClassifier("LR", bo_params={"base_estimator": "u"}, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


@pytest.mark.parametrize("callbacks", [TimerCallback(), [TimerCallback()]])
def test_callbacks(callbacks):
    """Assert that custom callbacks are accepted."""
    trainer = DirectClassifier(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"callbacks": callbacks},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)


def test_all_callbacks():
    """Assert that all callbacks predefined callbacks work as intended."""
    trainer = DirectClassifier(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"max_time": 50, "delta_x": 5, "delta_y": 5},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)


def test_invalid_max_time():
    """Assert than an error is raised when max_time<=0."""
    trainer = DirectClassifier("LR", bo_params={"max_time": 0}, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_invalid_delta_x():
    """Assert than an error is raised when delta_x<0."""
    trainer = DirectClassifier("LR", bo_params={"delta_x": -2}, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_invalid_delta_y():
    """Assert than an error is raised when delta_y<0."""
    trainer = DirectClassifier("LR", bo_params={"delta_y": -2}, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_plot():
    """Assert that plotting the BO runs without errors."""
    trainer = DirectClassifier(
        models=["lSVM", "kSVM", "MLP"],
        n_calls=35,
        n_initial_points=20,
        bo_params={"plot": True},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)


def test_invalid_cv():
    """Assert than an error is raised when cv<=0."""
    trainer = DirectClassifier("LR", bo_params={"cv": 0}, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_invalid_early_stopping():
    """Assert than an error is raised when early_stopping<=0."""
    trainer = DirectClassifier("LR", bo_params={"early_stopping": -1}, random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_custom_dimensions_all_models():
    """Assert that the custom dimensions are for all models if not dict."""
    trainer = DirectClassifier(
        models=["LR1", "LR2"],
        n_calls=2,
        n_initial_points=2,
        bo_params={"dimensions": [Integer(100, 1000, name="max_iter")]},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert list(trainer.lr1.best_params.keys()) == ["max_iter"]
    assert list(trainer.lr2.best_params.keys()) == ["max_iter"]


def test_custom_dimensions_per_model():
    """Assert that the custom dimensions are distributed over the models."""
    trainer = DirectClassifier(
        models=["LR1", "LR2"],
        n_calls=2,
        n_initial_points=2,
        bo_params={
            "dimensions": {
                "lr1": [Integer(100, 200, name="max_iter")],
                "lr2": [Integer(300, 400, name="max_iter")],
            },
        },
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert 100 <= trainer.lr1.best_params["max_iter"] <= 200
    assert 300 <= trainer.lr2.best_params["max_iter"] <= 400


def test_optimizer_kwargs():
    """Assert that the kwargs provided are passed to the optimizer."""
    trainer = DirectClassifier(
        models="LR",
        n_calls=2,
        n_initial_points=2,
        bo_params={"acq_func": "EI"},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer._bo_kwargs.get("acq_func") == "EI"


def test_est_params_all_models():
    """Assert that est_params passes the parameters to all models."""
    trainer = DirectClassifier(
        models=["XGB", "LGB"],
        n_calls=5,
        est_params={"n_estimators": 220},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.lgb.estimator.get_params()["n_estimators"] == 220
    assert trainer.xgb.estimator.get_params()["n_estimators"] == 220


def test_est_params_per_model():
    """Assert that est_params passes the parameters per model."""
    trainer = DirectClassifier(
        models=["XGB", "LGB"],
        est_params={"xgb": {"n_estimators": 100}, "lgb": {"n_estimators": 200}},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.xgb.estimator.get_params()["n_estimators"] == 100
    assert trainer.lgb.estimator.get_params()["n_estimators"] == 200


# Test _prepare_metric ============================================= >>

def test_invalid_sequence_parameter():
    """Assert that an error is raised for parameters with the wrong length."""
    trainer = DirectClassifier(
        models="LR",
        metric=f1_score,
        needs_proba=[True, False],
        random_state=1,
    )
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_metric_acronym():
    """"Assert that using the metric acronyms work."""
    trainer = DirectClassifier("LR", metric="auc", random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "roc_auc"


def test_invalid_scorer_name():
    """Assert that an error is raised when scorer name is invalid."""
    trainer = DirectClassifier("LR", metric="test", random_state=1)
    pytest.raises(ValueError, trainer.run, bin_train, bin_test)


def test_function_metric_parameter():
    """Assert that a function metric works."""
    trainer = DirectClassifier("LR", metric=f1_score, random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1_score"


def test_scorer_metric_parameter():
    """Assert that a scorer metric works."""
    trainer = DirectClassifier("LR", metric=get_scorer("f1"), random_state=1)
    trainer.run(bin_train, bin_test)
    assert trainer.metric == "f1"


# Test _core_iteration ============================================= >>

def test_sequence_parameters():
    """Assert that every model get his corresponding parameters."""
    trainer = DirectClassifier(
        models=["LR", "Tree", "LGB"],
        n_calls=(2, 3, 4),
        n_initial_points=(1, 2, 3),
        bagging=[2, 5, 7],
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert len(trainer.LR.bo) == 2
    assert sum(trainer.tree.bo.index.str.startswith("Initial")) == 2
    assert len(trainer.lgb.metric_bagging) == 7


def test_custom_dimensions_for_bo():
    """Assert that the BO runs when custom dimensions are provided."""
    trainer = DirectRegressor(
        models="OLS",
        n_calls=5,
        bo_params={"dimensions": [Categorical([True, False], name="fit_intercept")]},
        random_state=1,
    )
    trainer.run(reg_train, reg_test)
    assert not trainer.ols.bo.empty


def test_error_handling():
    """Assert that models with errors are removed from the pipeline."""
    trainer = DirectClassifier(
        models=["LR", "LDA"],
        n_calls=4,
        n_initial_points=[2, 5],
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert trainer.errors.get("LDA")
    assert "LDA" not in trainer.models
    assert "LDA" not in trainer.results.index


def test_close_plot_after_error():
    """Assert that the BO plot is closed after an error."""
    trainer = DirectClassifier(
        models=["LR", "LDA"],
        n_calls=4,
        n_initial_points=[2, 5],
        bo_params={"plot": True},
        random_state=1,
    )
    trainer.run(bin_train, bin_test)
    assert PlotCallback.c == 1  # First model is 0, after error passes to 1


def test_all_models_failed():
    """Assert that an error is raised when all models failed."""
    trainer = DirectClassifier("LR", n_calls=4, n_initial_points=5, random_state=1)
    pytest.raises(RuntimeError, trainer.run, bin_train, bin_test)
