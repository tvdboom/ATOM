"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for the models module.

"""

from platform import machine, system
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from optuna.distributions import IntDistribution
from optuna.pruners import PatientPruner
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sktime.forecasting.croston import Croston

from atom import ATOMClassifier, ATOMForecaster, ATOMModel, ATOMRegressor
from atom.pipeline import Pipeline

from .conftest import X_bin, X_class, X_ex, X_reg, y_bin, y_class, y_fc, y_reg


def test_custom_model_properties():
    """Assert that name and acronym are assigned correctly."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=ATOMModel(RandomForestRegressor(n_estimators=5, n_jobs=1)))
    assert atom.rfr.name == "RFR"
    assert atom.rfr.fullname == "RandomForestRegressor"


def test_custom_model_invalid_acronym():
    """Assert that an error is raised when name and acronym don't match."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(ValueError, match=".*do not match.*"):
        atom.run(models=ATOMModel(RandomForestRegressor, acronym="forest"))


def test_all_models_binary():
    """Assert that all models work with binary classification."""
    atom = ATOMClassifier(X_bin, y_bin, n_rows=0.5, random_state=1)
    atom.run(
        models=["!CatB", "!CatNB", "!LGB", "!RNN", "!XGB"],
        n_trials=5,
        est_params={
            "AdaB": {"n_estimators": 5},
            "Bag": {"n_estimators": 5},
            "ET": {"n_estimators": 5},
            "GBM": {"n_estimators": 5},
            "hGBM": {"max_iter": 5},
            "LR": {"max_iter": 5},
            "MLP": {"hidden_layer_sizes": (5,), "max_iter": 5},
            "PA": {"max_iter": 5},
            "Perc": {"max_iter": 5},
            "RF": {"n_estimators": 5},
            "SGD": {"max_iter": 5},
        },
        errors="raise",
    )


def test_all_models_multiclass():
    """Assert that all models work with multiclass classification."""
    atom = ATOMClassifier(X_class, y_class, n_rows=0.5, random_state=1)
    atom.run(
        models=["!CatB", "!CatNB", "!LGB", "!RNN", "!XGB"],
        n_trials=5,
        est_params={
            "AdaB": {"n_estimators": 5},
            "Bag": {"n_estimators": 5},
            "ET": {"n_estimators": 5},
            "GBM": {"n_estimators": 5},
            "hGBM": {"max_iter": 5},
            "KNN": {"n_neighbors": 2},
            "LR": {"max_iter": 5},
            "MLP": {"hidden_layer_sizes": (5,), "max_iter": 5},
            "PA": {"max_iter": 5},
            "Perc": {"max_iter": 5},
            "RF": {"n_estimators": 5},
            "SGD": {"max_iter": 5},
        },
        errors="raise",
    )


def test_all_models_regression():
    """Assert that all models work with regression."""
    atom = ATOMRegressor(X_reg, y_reg, n_rows=0.5, random_state=2)
    atom.run(
        models=["!CatB", "!LGB", "!RNN", "!XGB"],
        n_trials=5,
        est_params={
            "AdaB": {"n_estimators": 5},
            "Bag": {"n_estimators": 5},
            "ET": {"n_estimators": 5},
            "GBM": {"n_estimators": 5},
            "hGBM": {"max_iter": 5},
            "MLP": {"hidden_layer_sizes": (5,), "max_iter": 5},
            "PA": {"max_iter": 5},
            "RF": {"n_estimators": 5},
            "SGD": {"max_iter": 5},
        },
        errors="raise",
    )


def test_all_models_forecast():
    """Assert that all models work with forecast."""
    atom = ATOMForecaster(y_fc, sp=12, random_state=1)
    atom.run(
        models=["!DF", "!MSTL", "!VAR", "!VARMAX"],
        n_trials=1,
        est_params={
            "arima": {"maxiter": 5, "method": "nm"},
            "autoarima": {"maxiter": 5, "method": "nm"},
            "bats": {"use_box_cox": False, "use_trend": False, "use_arma_errors": False},
            "stl": {"seasonal": 3, "seasonal_deg": 0, "low_pass_deg": 0},
            "tbats": {"use_box_cox": False, "use_trend": False, "use_arma_errors": False},
            "ets": {"maxiter": 5},
        },
        errors="raise",
    )


def test_all_models_forecast_multivariate():
    """Assert that all models work with multivariate forecast."""
    atom = ATOMForecaster(X_ex, y=(-1, -2), random_state=1)
    atom.run(
        models=["DF", "VAR", "VARMAX"],
        n_trials=1,
        est_params={
            "df": {"k_factors": 1},
            "varmax": {"error_cov_type": "diagonal", "method": "nm", "maxiter": 5},
        },
        errors="raise",
    )


def test_univariate_models_forecast_custom_seasonality():
    """Assert that univariate models accept custom seasonality."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run(
        models=["ARIMA", "SARIMAX"],
        n_trials=1,
        est_params={
            "arima": {"maxiter": 5, "method": "nm"},
            "order": (1, 0, 0),
            "seasonal_order": (1, 0, 0, 12),
        },
        errors="raise",
    )


def test_multivariate_forecast_custom_seasonality():
    """Assert that multivariate models accept custom seasonality."""
    atom = ATOMForecaster(X_ex, y=(-1, -2), random_state=1)
    atom.run(
        models=["VARMAX"],
        n_trials=1,
        est_params={"error_cov_type": "diagonal", "method": "nm", "maxiter": 5, "order": (1, 0)},
        errors="raise",
    )


@pytest.mark.skipif(system() == "Darwin" or machine() not in ("x86_64", "AMD64"))
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_models_sklearnex_classification(device):
    """Assert the sklearnex engine works for classification tasks."""
    atom = ATOMClassifier(X_bin, y_bin, device=device, engine="sklearnex", random_state=1)
    atom.run(
        models=["KNN", "LR", "RF", "SVM"],
        n_trials=2,
        est_params={"LR": {"max_iter": 5}, "RF": {"n_estimators": 5}},
    )
    assert all(m.estimator.__module__.startswith(("daal4py", "sklearnex")) for m in atom._models)


@pytest.mark.skipif(machine() not in ("x86_64", "AMD64"), reason="Only x86 support.")
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_models_sklearnex_regression(device):
    """Assert the sklearnex engine works for regression tasks."""
    atom = ATOMRegressor(X_reg, y_reg, device=device, engine="sklearnex", random_state=1)
    atom.run(
        models=["EN", "KNN", "Lasso", "OLS", "RF", "Ridge", "SVM"],
        n_trials=2,
        est_params={"RF": {"n_estimators": 5}},
    )
    assert all(m.estimator.__module__.startswith(("daal4py", "sklearnex")) for m in atom._models)


@patch.dict(
    "sys.modules",
    {
        "cuml": Mock(spec=["__spec__"]),
        "cuml.common.device_selection": Mock(spec=["set_global_device_type"]),
        "cuml.internals.memory_utils": Mock(spec=["set_global_output_type"]),
    },
)
def test_models_cuml_classification():
    """Assert that all classification models can be called with cuml."""
    atom = ATOMClassifier(X_bin, y_bin, device="gpu", engine="cuml", verbose=2, random_state=1)
    atom.run(
        models=["!CatB", "!LGB", "!XGB"],
        n_trials=1,
        est_params={
            "AdaB": {"n_estimators": 5},
            "Bag": {"n_estimators": 5},
            "ET": {"n_estimators": 5},
            "GBM": {"n_estimators": 5},
            "hGBM": {"max_iter": 5},
            "LR": {"max_iter": 5},
            "MLP": {"hidden_layer_sizes": (5,), "max_iter": 5},
            "PA": {"max_iter": 5},
            "Perc": {"max_iter": 5},
            "RF": {"n_estimators": 5},
            "SGD": {"max_iter": 5},
        },
    )


@patch.dict(
    "sys.modules",
    {
        "cuml": Mock(spec=["__spec__"]),
        "cuml.common.device_selection": Mock(spec=["set_global_device_type"]),
        "cuml.internals.memory_utils": Mock(spec=["set_global_output_type"]),
    },
)
def test_models_cuml_regression():
    """Assert that all regression models can be called with cuml."""
    atom = ATOMRegressor(X_reg, y_reg, device="gpu", engine="cuml", random_state=1)
    atom.run(
        models=["!CatB", "!LGB", "!XGB"],
        n_trials=1,
        est_params={
            "AdaB": {"n_estimators": 5},
            "Bag": {"n_estimators": 5},
            "ET": {"n_estimators": 5},
            "GBM": {"n_estimators": 5},
            "hGBM": {"max_iter": 5},
            "MLP": {"hidden_layer_sizes": (5,), "max_iter": 5},
            "PA": {"max_iter": 5},
            "RF": {"n_estimators": 5},
            "SGD": {"max_iter": 5},
        },
    )


def test_CatNB(random):
    """Assert that the CatNB model works. Needs special dataset."""
    X = random.integers(2, size=(150, 10))
    y = random.integers(2, size=150)

    atom = ATOMClassifier(X, y, random_state=1)
    assert atom.scaled  # Check scaling is True for all binary columns
    atom.run(models="CatNB", n_trials=1)
    assert hasattr(atom, "CatNB")


def test_RNN():
    """Assert that the RNN model works. Fails with default parameters."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("RNN", n_trials=1, est_params={"outlier_label": "most_frequent"})
    assert atom.models == "RNN"


@patch("sktime.forecasting.statsforecast.StatsForecastMSTL")
def test_MSTL_with_stl_kwargs_params(cls):
    """Assert that the MSTL model works when providing stl_kwargs params."""
    cls.return_value.fit = Mock()
    cls.return_value.set_params.return_value.predict.__name__ = "predict"

    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run(
        models="MSTL",
        n_trials=1,
        est_params={"stl_kwargs": {"seasonal_deg": 0}},
        ht_params={"catch": (Exception,)},
        errors="raise",
    )
    assert atom.models == "MSTL"


def test_Prophet_non_standard_seasonality():
    """Assert that the Prophet model works with non-standard seasonality."""
    atom = ATOMForecaster(y_fc, sp=3, random_state=1)
    atom.run("Prophet")
    assert atom.models == "Prophet"


@pytest.mark.parametrize("model", ["CatB", "LGB", "XGB"])
def test_pruning_non_sklearn(model):
    """Assert that non-sklearn models can be pruned."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(
        models=model,
        n_trials=7,
        est_params={"n_estimators": 10, "max_depth": 2},
        ht_params={"pruner": PatientPruner(None, patience=1)},
    )
    assert "PRUNED" in atom.winner.trials["state"].unique()


@pytest.mark.parametrize("model", ["CatB", "LGB", "XGB"])
def test_custom_metric_evals_non_sklearn(model):
    """Assert that non-sklearn models can be pruned."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(model, est_params={"n_estimators": 5, "max_depth": 2})
    assert getattr(atom, model).evals is not None

    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(model, est_params={"n_estimators": 5, "max_depth": 2})
    assert getattr(atom, model).evals is not None

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(model, est_params={"n_estimators": 5, "max_depth": 2})
    assert getattr(atom, model).evals is not None


def test_MLP_custom_hidden_layer_sizes():
    """Assert that the MLP model can have custom hidden_layer_sizes."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="MLP",
        n_trials=1,
        est_params={"hidden_layer_sizes": (31, 2), "max_iter": 5},
    )
    assert "hidden_layer_1" not in atom.mlp.best_params
    assert atom.mlp.estimator.get_params()["hidden_layer_sizes"] == (31, 2)


def test_MLP_custom_n_layers():
    """Assert that the MLP model can have a custom number of hidden layers."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models="MLP",
        n_trials=5,
        est_params={"max_iter": 5},
        ht_params={
            "distributions": {
                "hidden_layer_1": IntDistribution(2, 4),
                "hidden_layer_2": IntDistribution(0, 4),
                "hidden_layer_3": IntDistribution(0, 4),
                "hidden_layer_4": IntDistribution(0, 4),
            },
        },
    )
    assert "hidden_layer_1" in atom.mlp.trials


# Test ensembles =================================================== >>

def test_ensemble_failed_feature_importance():
    """Assert that the Stacking model works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(models=["LDA", "SVM"])
    atom.voting()
    with pytest.raises(ValueError, match=".*feature importance for meta-estimator.*"):
        print(atom.vote.feature_importance)


def test_stacking():
    """Assert that the Stacking model works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(models=["RF", "XGB"])
    atom.stacking()
    assert isinstance(atom.stack.estimator.estimators_[0], RandomForestRegressor)
    assert isinstance(atom.stack.estimator.estimators_[1], Pipeline)
    assert isinstance(atom.stack.feature_importance, pd.Series)


def test_stacking_forecast():
    """Assert that the Stacking model works for forecast tasks."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run(models=["Croston", "Theta"])
    atom.stacking()
    assert isinstance(atom.stack.estimator.forecasters_[0], Croston)


def test_voting():
    """Assert that the Voting model works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(
        models=["SVM", "RF", "XGB"],
        est_params={
            "RF": {"n_estimators": 5, "max_depth": 2},
            "XGB": {"n_estimators": 5, "max_depth": 2},
        },
    )

    # Not all models have predict_proba
    with pytest.raises(ValueError, match=".*a predict_proba method.*"):
        atom.voting(voting="soft")

    atom.voting(models=["RF", "XGB"])
    assert isinstance(atom.vote.estimator.estimators_[0], RandomForestClassifier)
    assert isinstance(atom.vote.estimator.estimators_[1], Pipeline)
    assert isinstance(atom.vote.feature_importance, pd.Series)


def test_voting_forecast():
    """Assert that the Voting model works for forecast tasks."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run(models=["Croston", "Theta"])
    atom.voting()
    assert isinstance(atom.vote.estimator.forecasters_[0], Croston)
