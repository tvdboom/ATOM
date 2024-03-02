"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Unit tests for basetransformer.py

"""

import multiprocessing
import os
from logging import Logger
from pathlib import Path
from platform import machine
from random import sample
from unittest.mock import patch

import mlflow
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearnex import get_config
from sklearnex.svm import SVC

from atom import ATOMClassifier, ATOMForecaster
from atom.basetransformer import BaseTransformer
from atom.training import DirectClassifier
from atom.utils.types import EngineTuple
from atom.utils.utils import merge

from .conftest import (
    X10, X_bin, X_bin_array, X_sparse, X_text, bin_test, bin_train, y10, y_bin,
    y_bin_array, y_fc,
)


# Test properties ================================================== >>

def test_n_jobs_maximum_cores():
    """Assert that value equals n_cores if maximum is exceeded."""
    base = BaseTransformer(n_jobs=1000)
    assert base.n_jobs == multiprocessing.cpu_count()


def test_negative_n_jobs():
    """Assert that value is set correctly for negative values."""
    base = BaseTransformer(n_jobs=-1)
    assert base.n_jobs == multiprocessing.cpu_count()

    base = BaseTransformer(n_jobs=-2)
    assert base.n_jobs == multiprocessing.cpu_count() - 1


def test_device_parameter():
    """Assert that the device is set correctly."""
    BaseTransformer(device="gpu")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize("engine", [None, "pandas", "sklearn", {}, EngineTuple()])
def test_engine_parameter(engine):
    """Assert that the engine parameter can be initialized."""
    base = BaseTransformer(engine=engine)
    assert base.engine == EngineTuple()


@pytest.mark.skipif(machine() not in ("x86_64", "AMD64"), reason="Only x86 support")
def test_engine_parameter_sklearnex():
    """Assert that sklearnex offloads to the right device."""
    BaseTransformer(device="gpu", engine={"estimator": "sklearnex"})
    assert get_config()["target_offload"] == "gpu"


def test_engine_parameter_no_cuml():
    """Assert that an error is raised when cuml is not installed."""
    with pytest.raises(ModuleNotFoundError, match=".*Failed to import cuml.*"):
        BaseTransformer(device="gpu", engine={"estimator": "cuml"})


@patch("ray.init")
def test_backend_parameter_ray(ray):
    """Assert that ray is initialized when selected."""
    BaseTransformer(backend="ray")
    assert ray.is_called_once


@patch("dask.distributed.Client")
def test_backend_parameter_dask(dask):
    """Assert that dask is initialized when selected."""
    BaseTransformer(backend="dask")
    assert dask.is_called_once


def test_backend_parameter():
    """Assert that other backends can be specified."""
    base = BaseTransformer(backend="threading")
    assert base.backend == "threading"


def test_warnings_parameter_bool():
    """Assert that the warnings parameter works for a bool."""
    base = BaseTransformer(warnings=True)
    assert base.warnings == "once"

    base = BaseTransformer(warnings=False)
    assert base.warnings == "ignore"


def test_warnings_parameter_str():
    """Assert that the warnings parameter works for a str."""
    base = BaseTransformer(warnings="always")
    assert base.warnings == "always"


@pytest.mark.parametrize("logger", [None, "auto", Path("test"), Logger("test")])
def test_logger_creator(logger):
    """Assert that the logger is created correctly."""
    base = BaseTransformer(logger=logger)
    assert isinstance(base.logger, Logger | None)


def test_crash_with_logger():
    """Assert that the crash decorator works with a logger."""
    atom = ATOMClassifier(X_bin, y_bin, logger="log")
    pytest.raises(RuntimeError, atom.run, "LR", est_params={"test": 2})
    with open("log.log", "rb") as f:
        assert "raise RuntimeError" in str(f.read())


@patch("mlflow.set_experiment")
def test_experiment_creation(mlflow):
    """Assert that the mlflow experiment is created."""
    base = BaseTransformer(experiment="test")
    assert base.experiment == "test"
    mlflow.assert_called_once()


@patch("dagshub.auth.get_token")
@patch("requests.get")
@patch("dagshub.init")
def test_experiment_dagshub(dagshub, request, token):
    """Assert that the experiment can be stored in dagshub."""
    token.return_value = "token"
    request.return_value.text = {"username": "user1"}

    BaseTransformer(experiment="dagshub:test")
    dagshub.assert_called_once()
    assert "dagshub" in mlflow.get_tracking_uri()

    # Reset to default URI
    BaseTransformer(experiment="test")
    assert "dagshub" not in mlflow.get_tracking_uri()


def test_device_id_no_value():
    """Assert that the device id can be left empty."""
    base = BaseTransformer(device="gpu")
    assert base._device_id == 0


def test_device_id_int():
    """Assert that the device id can be set."""
    base = BaseTransformer(device="gpu:2")
    assert base._device_id == 2


def test_device_id_invalid():
    """Assert that an error is raised when the device id is invalid."""
    with pytest.raises(ValueError, match=".*Use a single integer.*"):
        BaseTransformer(device="gpu:2,3")


# Test _check_input ============================================== >>

def test_input_is_copied():
    """Assert that the data is copied."""
    X, y = BaseTransformer._check_input(X_bin, y_bin)
    assert X is not X_bin
    assert y is not y_bin


def test_input_X_and_y_None():
    """Assert that an error is raised when both X and y are None."""
    with pytest.raises(ValueError, match=".*both None.*"):
        BaseTransformer._check_input()


def test_input_is_numpy():
    """Assert that the data provided is converted to pandas objects."""
    X, y = BaseTransformer._check_input(X_bin_array, y_bin_array)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_input_is_polars():
    """Assert that the data provided can be a callable."""
    X, y = BaseTransformer._check_input(pl.from_pandas(X_bin), pl.from_pandas(y_bin))
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_X_is_callable():
    """Assert that the data provided can be a callable."""
    X, _ = BaseTransformer._check_input(lambda: [[1, 2], [2, 1], [3, 1]])
    assert isinstance(X, pd.DataFrame)


def test_column_order_is_retained():
    """Assert that column order is kept if column names are specified."""
    X_shuffled = X_bin[sample(list(X_bin.columns), X_bin.shape[1])]
    X, _ = BaseTransformer._check_input(X_shuffled, columns=X_bin.columns)
    assert list(X.columns) == list(X_bin.columns)


def test_incorrect_columns():
    """Assert that an error is raised when the provided columns do not match."""
    with pytest.raises(ValueError, match=".*columns are different.*"):
        BaseTransformer._check_input(X_bin, columns=["1", "2"])


def test_input_data_in_atom():
    """Assert that the data does not change once in an atom pipeline."""
    atom = ATOMClassifier(X10, y10, random_state=1)
    X10[3][2] = 99  # Change an item of the original variable
    assert 99 not in atom.dataset  # Is unchanged in the pipeline


def test_input_data_in_training():
    """Assert that the data does not change once in a training pipeline."""
    train = bin_train.copy()
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(train, bin_test)
    train.iloc[3, 2] = 99  # Change an item of the original variable
    assert 99 not in trainer.dataset  # Is unchanged in the pipeline


def test_text_to_corpus():
    """Assert that for text data the column is named corpus."""
    atom = ATOMClassifier(X_text, y10, random_state=1)
    assert atom.X.columns == ["corpus"]


def test_int_columns_to_str():
    """Assert that int columns are converted to str."""
    X = X_bin.copy()
    X.columns = range(X.shape[1])
    atom = ATOMClassifier(X, y_bin, random_state=1)
    assert atom.X.columns[0] == "0"


def test_error_multiindex():
    """Assert that an error is raised for multiindex dataframes."""
    X = X_bin.copy()
    X.columns = pd.MultiIndex.from_product([["dummy"], X.columns])
    with pytest.raises(ValueError, match=".*MultiIndex columns are not supported.*"):
        ATOMClassifier(X, y_bin, random_state=1)


def test_duplicate_column_names_in_X():
    """Assert that an error is raised when X has duplicate column names."""
    X = merge(X_bin.copy(), pd.Series(1, name="mean texture"))
    with pytest.raises(ValueError, match=".*column names found in X.*"):
        ATOMClassifier(X, y_bin, random_state=1)


def test_sparse_matrices_X_y():
    """Assert that sparse matrices are accepted as (X, y) input."""
    atom = ATOMClassifier(X_sparse, y10, random_state=1)
    assert isinstance(atom.X, pd.DataFrame)
    assert atom.shape == (10, 4)
    assert atom[atom.columns[0]].dtype.name == "Sparse[int64, 0]"


def test_sparse_matrices_2_tuples():
    """Assert that sparse matrices are accepted as 2-tuples input."""
    atom = ATOMClassifier((X_sparse, y10), (X_sparse, y10), random_state=1)
    assert isinstance(atom.X, pd.DataFrame)
    assert atom.shape == (20, 4)
    assert atom[atom.columns[0]].dtype.name == "Sparse[int64, 0]"


def test_multioutput_str():
    """Assert that multioutput can be assigned by column name."""
    X, y = BaseTransformer._check_input(X_bin, ["mean radius", "worst perimeter"])
    assert list(y.columns) == ["mean radius", "worst perimeter"]


def test_multioutput_int():
    """Assert that multioutput can be assigned by column position."""
    X, y = BaseTransformer._check_input(X_bin, [0, 2])
    assert list(y.columns) == ["mean radius", "mean perimeter"]


def test_equal_length():
    """Assert that an error is raised when X and y have unequal length."""
    with pytest.raises(ValueError, match=".*number of rows.*"):
        BaseTransformer._check_input(X10, [312, 22])


def test_equal_index():
    """Assert that an error is raised when X and y don't have same indices."""
    y = pd.Series(y_bin_array, index=range(10, len(y_bin_array) + 10))
    with pytest.raises(ValueError, match=".*same indices.*"):
        BaseTransformer._check_input(X_bin, y)


def test_target_is_string():
    """Assert that the target column is assigned correctly for a string."""
    _, y = BaseTransformer._check_input(X_bin, y="mean radius")
    assert y.name == "mean radius"


def test_target_not_in_dataset():
    """Assert that the target column given by y is in X."""
    with pytest.raises(ValueError, match=".*not found in X.*"):
        BaseTransformer._check_input(X_bin, "X")


def test_X_is_None_with_str():
    """Assert that an error is raised when X is None and y is a string."""
    with pytest.raises(ValueError, match=".*can't be None when y is a str.*"):
        BaseTransformer._check_input(y="test")


def test_target_is_int():
    """Assert that target column is assigned correctly for an integer."""
    _, y = BaseTransformer._check_input(X_bin, y=0)
    assert y.name == "mean radius"


def test_target_is_dict():
    """Assert that target column is assigned correctly for a dictionary."""
    _, y = BaseTransformer._check_input(X10, {"y1": y10, "y2": y10})
    assert list(y.columns) == ["y1", "y2"]


def test_X_is_None_with_int():
    """Assert that an error is raised when X is None and y is an int."""
    with pytest.raises(ValueError, match=".*can't be None when y is an int.*"):
        BaseTransformer._check_input(y=1)


def test_target_is_none():
    """Assert that target column stays None when empty input."""
    _, y = BaseTransformer._check_input(X_bin, y=None)
    assert y is None


def test_X_empty_df():
    """Assert that X becomes an empty dataframe when provided but in y."""
    X, y = BaseTransformer._check_input(y_fc, y=-1)
    assert X.empty
    assert isinstance(y, pd.Series)


# Test _inherit ==================================================== >>

def test_inherit():
    """Assert that the inherit method passes the parameters correctly."""
    base = BaseTransformer(n_jobs=2, random_state=2)
    svc = base._inherit(RandomForestClassifier())
    assert svc.get_params()["n_jobs"] == 2
    assert svc.get_params()["random_state"] == 2


def test_inherit_with_fixed_params():
    """Assert that fixed parameters aren't inherited."""
    base = BaseTransformer(random_state=2)
    chain = base._inherit(ClassifierChain(SVC(random_state=3)), ("base_estimator__random_state",))
    assert chain.get_params()["random_state"] == 2
    assert chain.base_estimator.get_params()["random_state"] == 3


def test_inherit_sp():
    """Assert that the seasonal periodicity is correctly inherited."""
    atom = ATOMForecaster(y_fc, sp=[12, 24], random_state=1)
    atom.run(
        models=["bats", "tbats"],
        est_params={
            "bats": {"use_box_cox": False, "use_trend": False, "use_arma_errors": False},
            "tbats": {"use_box_cox": False, "use_trend": False, "use_arma_errors": False},
        },
    )
    assert atom.bats.estimator.get_params()["sp"] == 12  # Single seasonality
    assert atom.tbats.estimator.get_params()["sp"] == [12, 24]  # Multiple seasonality


# Test _get_est_class ============================================== >>

@pytest.mark.skipif(machine() not in ("x86_64", "AMD64"), reason="Only x86 support")
def test_get_est_class_from_engine():
    """Assert that the class can be retrieved from an engine."""
    base = BaseTransformer(device="cpu", engine={"estimator": "sklearnex"})
    assert base._get_est_class("SVC", "svm") == SVC


def test_get_est_class_from_default():
    """Assert that the class is retrieved from sklearn when import fails."""
    base = BaseTransformer(device="cpu", engine={"estimator": "sklearnex"})
    assert base._get_est_class("GaussianNB", "naive_bayes") == GaussianNB


# Test log ========================================================= >>

def test_log_severity_error():
    """Assert that an error is raised when the severity is error."""
    with pytest.raises(UserWarning, match=".*user error.*"):
        BaseTransformer(logger="log")._log("this is a user error", severity="error")


@patch("atom.basetransformer.getLogger")
def test_log(cls):
    """Assert the log method works."""
    base = BaseTransformer(verbose=2, logger="log")
    base._log("test", 1)
    cls.return_value.info.assert_called()
