# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the BaseTransformer class.

"""

from __future__ import annotations

import os
import random
import tempfile
import warnings
from copy import deepcopy
from datetime import datetime as dt
from importlib import import_module
from importlib.util import find_spec
from logging import DEBUG, FileHandler, Formatter, Logger, getLogger
from multiprocessing import cpu_count
from typing import Callable

import dagshub
import mlflow
import numpy as np
import ray
import requests
from dagshub.auth.token_auth import HTTPBearerAuth
from joblib.memory import Memory
from ray.util.joblib import register_ray
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_memory
from sktime.datatypes import check_is_mtype

from atom.utils.types import (
    Backend, Bool, DataFrame, DataFrameTypes, Engine, Estimator, Features, Int,
    IntTypes, Pandas, Scalar, Sequence, SequenceTypes, Severity, Target,
    Verbose, Warnings,
)
from atom.utils.utils import (
    DataContainer, bk, crash, get_cols, lst, merge, n_cols, pd, sign, to_df,
    to_pandas,
)


class BaseTransformer:
    """Base class for transformers in the package.

    Note that this includes atom and runners. Contains shared
    properties, data preparation methods and utility methods.

    Parameters
    ----------
    **kwargs
        Standard keyword arguments for the classes. Can include:

        - n_jobs: Number of cores to use for parallel processing.
        - device: Device on which to run the estimators.
        - engine: Execution engine to use for data and estimators.
        - backend: Parallelization backend.
        - verbose: Verbosity level of the output.
        - warnings: Whether to show or suppress encountered warnings.
        - logger: Name of the log file, Logger object or None.
        - experiment: Name of the mlflow experiment used for tracking.
        - random_state: Seed used by the random number generator.

    """

    attrs = [
        "n_jobs",
        "device",
        "engine",
        "backend",
        "memory",
        "verbose",
        "warnings",
        "logger",
        "experiment",
        "random_state",
    ]

    def __init__(self, **kwargs):
        """Update the properties with the provided kwargs."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Properties =================================================== >>

    @property
    def n_jobs(self) -> Int:
        """Number of cores to use for parallel processing."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: Int):
        # Check number of cores for multiprocessing
        if value > (n_cores := cpu_count()):
            value = n_cores
        else:
            value = n_cores + 1 + value if value < 0 else value

            # Final check for negative input
            if value < 1:
                raise ValueError(
                    "Invalid value for the n_jobs parameter, "
                    f"got {value}. Value should be >=0.", 1
                )

        self._n_jobs = value

    @property
    def device(self) -> str:
        """Device on which to run the estimators."""
        return self._device

    @device.setter
    def device(self, value: str):
        self._device = value
        if "gpu" in value.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._device_id)

    @property
    def engine(self) -> Engine:
        """Execution engine for estimators."""
        return self._engine

    @engine.setter
    def engine(self, value: Engine):
        if value.get("data") == "modin" and not ray.is_initialized():
            ray.init(
                runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_Pandas__": "1"}},
                log_to_driver=False,
            )

        # Update env variable to use for PandasModin in utils.py
        os.environ["ATOM_DATA_Engine"] = value.get("data", "numpy")

        if value.get("estimator") == "sklearnex":
            if not find_spec("sklearnex"):
                raise ModuleNotFoundError(
                    "Failed to import scikit-learn-intelex. The library is "
                    "not installed. Note that the library only supports CPUs "
                    "with a x86 architecture."
                )
            else:
                import sklearnex
                sklearnex.set_config(self.device.lower() if self._gpu else "auto")
        elif value.get("estimator") == "cuml":
            if not find_spec("cuml"):
                raise ModuleNotFoundError(
                    "Failed to import cuml. Package is not installed. Refer "
                    "to: https://rapids.ai/start.html#install."
                )
            else:
                from cuml.common.device_selection import set_global_device_type
                set_global_device_type("gpu" if self._gpu else "cpu")

                # See https://github.com/rapidsai/cuml/issues/5564
                from cuml.internals.memory_utils import set_global_output_type
                set_global_output_type("numpy")

        self._engine = value

    @property
    def backend(self) -> Backend:
        """Parallelization backend."""
        return self._backend

    @backend.setter
    def backend(self, value: Backend):
        if value == "ray":
            register_ray()  # Register ray as joblib backend
            if not ray.is_initialized():
                ray.init(log_to_driver=False)

        self._backend = value

    @property
    def memory(self) -> Memory:
        """Get the internal memory object."""
        return self._memory

    @memory.setter
    def memory(self, value: Bool | str | Memory):
        """Create a new internal memory object."""
        if value is False:
            value = None
        elif value is True:
            value = tempfile.gettempdir()

        self._memory = check_memory(value)

    @property
    def verbose(self) -> Verbose:
        """Verbosity level of the output."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: Verbose):
        self._verbose = value

    @property
    def warnings(self) -> Warnings:
        """Whether to show or suppress encountered warnings."""
        return self._warnings

    @warnings.setter
    def warnings(self, value: Bool | Warnings):
        if isinstance(value, Bool):
            self._warnings = "default" if value else "ignore"
        else:
            self._warnings = value

        warnings.filterwarnings(self._warnings)  # Change the filter in this process
        warnings.filterwarnings("ignore", category=FutureWarning, module=".*pandas.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module=".*imblearn.*")
        warnings.filterwarnings("ignore", category=UserWarning, module=".*sktime.*")
        warnings.filterwarnings("ignore", category=ResourceWarning, module=".*ray.*")
        warnings.filterwarnings("ignore", category=UserWarning, module=".*modin.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*shap.*")
        os.environ["PYTHONWarnings"] = self._warnings  # Affects subprocesses (joblib)

    @property
    def logger(self) -> Logger | None:
        """Logger for this instance."""
        return self._logger

    @logger.setter
    def logger(self, value: str | Logger | None):
        external_loggers = [
            "dagshub",
            "mlflow",
            "optuna",
            "ray",
            "modin",
            "featuretools",
            "evalml",
        ]

        # Clear existing handlers for external loggers
        for name in external_loggers:
            for handler in (log := getLogger(name)).handlers:
                handler.close()
            log.handlers.clear()

        if not value:
            logger = None
        else:
            if isinstance(value, Logger):
                logger = value
            else:
                logger = getLogger(self.__class__.__name__)
                logger.setLevel(DEBUG)

                # Clear existing handlers for current logger
                for handler in logger.handlers:
                    handler.close()
                logger.handlers.clear()

                # Prepare the FileHandler
                if not value.endswith(".log"):
                    value += ".log"
                if os.path.basename(value) == "auto.log":
                    current = dt.now().strftime("%d%b%y_%Hh%Mm%Ss")
                    value = value.replace("auto", self.__class__.__name__ + "_" + current)

                fh = FileHandler(value)
                fh.setFormatter(Formatter("%(asctime)s - %(levelname)s: %(message)s"))

                # Redirect loggers to file handler
                for log in [logger.name] + external_loggers:
                    getLogger(log).addHandler(fh)

        self._logger = logger

    @property
    def experiment(self) -> str | None:
        """Name of the mlflow experiment used for tracking."""
        return self._experiment

    @experiment.setter
    def experiment(self, value: str | None):
        self._experiment = value
        if value:
            if value.lower().startswith("dagshub:"):
                value = value[8:]  # Drop dagshub:

                token = dagshub.auth.get_token()
                os.environ["MLFLOW_TRACKING_USERNAME"] = token
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token

                # Fetch username from dagshub api
                username = requests.get(
                    url="https://dagshub.com/api/v1/user",
                    auth=HTTPBearerAuth(token),
                ).json()["username"]

                if f"{username}/{value}" not in os.getenv("MLFLOW_TRACKING_URI", ""):
                    dagshub.init(repo_name=value, repo_owner=username)
                    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

            elif "dagshub" in mlflow.get_tracking_uri():
                mlflow.set_tracking_uri("")  # Reset URI to ./mlruns

            mlflow.sklearn.autolog(disable=True)
            mlflow.set_experiment(value)

    @property
    def random_state(self) -> Int | None:
        """Seed used by the random number generator."""
        return self._random_state

    @random_state.setter
    def random_state(self, value: Int | None):
        if value and value < 0:
            raise ValueError(
                "Invalid value for the random_state parameter. "
                f"Value should be >0, got {value}."
            )
        random.seed(value)
        np.random.seed(value)
        self._random_state = value

    @property
    def _gpu(self) -> Bool:
        """Return whether the instance uses a GPU implementation."""
        return "gpu" in self.device.lower()

    @property
    def _device_id(self) -> int:
        """Which GPU device to use."""
        if len(value := self.device.split(":")) == 1:
            return 0  # Default value
        else:
            try:
                return int(value[-1])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid value for the device parameter. GPU device {value[-1]} "
                    "isn't understood. Use a single integer to denote a specific "
                    "device. Note that ATOM doesn't support multi-GPU training."
                )

    # Methods ====================================================== >>

    def _inherit(self, obj: Estimator) -> Estimator:
        """Inherit n_jobs and/or random_state from parent.

        Utility method to set the n_jobs and random_state parameters
        of an estimator (if available) equal to that of this instance.

        Parameters
        ----------
        obj: object
            Object in which to change the parameters.

        Returns
        -------
        object
            Same object with changed parameters.

        """
        signature = sign(obj.__init__)
        for p in ("n_jobs", "random_state"):
            if p in signature and getattr(obj, p, "<!>") == signature[p]._default:
                setattr(obj, p, getattr(self, p))

        return obj

    def _get_est_class(self, name: str, module: str) -> Estimator:
        """Import a class from a module.

        When the import fails, for example if atom uses sklearnex and
        that's passed to a transformer, use sklearn's (default engine).

        Parameters
        ----------
        name: str
            Name of the class to get.

        module: str
            Module from which to get the class.

        Returns
        -------
        Estimator
            Class of the estimator.

        """
        try:
            engine = self.engine.get("estimator", "sklearn")
            return getattr(import_module(f"{engine}.{module}"), name)
        except (ModuleNotFoundError, AttributeError):
            return getattr(import_module(f"sklearn.{module}"), name)

    @staticmethod
    def _prepare_input(
        X: Callable | Features | None = None,
        y: Target | None = None,
        columns: Sequence | None = None,
    ) -> tuple[DataFrame | None, Pandas | None]:
        """Prepare the input data.

        Convert X and y to pandas (if not already) and perform standard
        compatibility checks (dimensions, length, indices, etc...).

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence,dataframe or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        columns: sequence or None
            Names of the features corresponding to X. If X already is a
            dataframe, force feature order. If None and X is not a
            dataframe, assign default feature names.

        Returns
        -------
        dataframe or None
            Feature dataset. Only returned if provided.

        series, dataframe or None
            Target column corresponding to X.

        """
        if X is None and y is None:
            raise ValueError("X and y can't be both None!")
        elif X is not None:
            X = to_df(deepcopy(X() if callable(X) else X), columns=columns)

            # If text dataset, change the name of the column to corpus
            if list(X.columns) == ["x0"] and X[X.columns[0]].dtype == "object":
                X = X.rename(columns={X.columns[0]: "corpus"})
            else:
                # Convert all column names to str
                X.columns = map(str, X.columns)

                # No duplicate rows nor column names are allowed
                if X.columns.duplicated().any():
                    raise ValueError("Duplicate column names found in X.")

                # Reorder columns to original order
                if columns is not None:
                    try:
                        X = X[list(columns)]  # Force feature order determined by columns
                    except KeyError:
                        raise ValueError("Features are different than seen at fit time.")

        # Prepare target column
        if isinstance(y, (dict, *SequenceTypes, *DataFrameTypes)):
            if isinstance(y, dict):
                if n_cols(y := to_df(deepcopy(y), index=getattr(X, "index", None))) == 1:
                    y = y.iloc[:, 0]  # If y is one-dimensional, get series

            elif isinstance(y, (*SequenceTypes, *DataFrameTypes)):
                # If X and y have different number of rows, try multioutput
                if X is not None and len(X) != len(y):
                    try:
                        targets = []
                        for col in y:
                            if col in X.columns:
                                targets.append(col)
                            else:
                                targets.append(X.columns[col])

                        X, y = X.drop(targets, axis=1), X[targets]

                    except (TypeError, IndexError, KeyError):
                        raise ValueError(
                            "X and y don't have the same number of rows,"
                            f" got len(X)={len(X)} and len(y)={len(y)}."
                        ) from None

                y = to_pandas(
                    data=deepcopy(y),
                    index=getattr(X, "index", None),
                    columns=[f"y{i}" for i in range(n_cols(y))],
                )

            # Check X and y have the same indices
            if X is not None and not X.index.equals(y.index):
                raise ValueError("X and y don't have the same indices!")

        elif isinstance(y, str):
            if X is not None:
                if y not in X.columns:
                    raise ValueError(f"Column {y} not found in X!")

                X, y = X.drop(y, axis=1), X[y]

            else:
                raise ValueError("X can't be None when y is a string.")

        elif isinstance(y, IntTypes):
            if X is None:
                raise ValueError("X can't be None when y is an int.")

            X, y = X.drop(X.columns[y], axis=1), X[X.columns[y]]

        return X, y

    def _set_index(self, df: DataFrame, y: Pandas | None) -> DataFrame:
        """Assign an index to the dataframe.

        Parameters
        ----------
        df: dataframe
            Dataset.

        y: series, dataframe or None
            Target column(s). Used to check that the provided index
            is not one of the target columns. If None, the check is
            skipped.

        Returns
        -------
        dataframe
            Dataset with updated indices.

        """
        if self._config.index is True:  # True gets caught by isinstance(int)
            pass
        elif self._config.index is False:
            df = df.reset_index(drop=True)
        elif isinstance(self._config.index, IntTypes):
            if -df.shape[1] <= self._config.index <= df.shape[1]:
                df = df.set_index(df.columns[self._config.index], drop=True)
            else:
                raise IndexError(
                    f"Invalid value for the index parameter. Value {self._config.index} "
                    f"is out of range for a dataset with {df.shape[1]} columns."
                )
        elif isinstance(self._config.index, str):
            if self._config.index in df:
                df = df.set_index(self._config.index, drop=True)
            else:
                raise ValueError(
                    "Invalid value for the index parameter. "
                    f"Column {self._config.index} not found in the dataset."
                )

        if y is not None and df.index.name in (c.name for c in get_cols(y)):
            raise ValueError(
                "Invalid value for the index parameter. The index column "
                f"can not be the same as the target column, got {df.index.name}."
            )

        if df.index.duplicated().any():
            raise ValueError(
                "Invalid value for the index parameter. There are duplicate indices "
                "in the dataset. Use index=False to reset the index to RangeIndex."
            )

        return df

    def _get_stratify_columns(self, df: DataFrame, y: Pandas) -> DataFrame | None:
        """Get columns to stratify by.

        Parameters
        ----------
        df: dataframe
            Dataset from which to get the columns.

        y: series or dataframe
            Target column.

        Returns
        -------
        dataframe or None
            Dataset with subselection of columns. Returns None if
            there's no stratification.

        """
        # Stratification is not possible when the data cannot change order
        if self._config.stratify is False:
            return None
        elif self._config.shuffle is False:
            self._log(
                "Stratification is not possible when shuffle=False.", 3,
                severity="warning"
            )
            return None
        elif self._config.stratify is True:
            return df[[c.name for c in get_cols(y)]]
        else:
            inc = []
            for col in lst(self._config.stratify):
                if isinstance(col, IntTypes):
                    if -df.shape[1] <= col <= df.shape[1]:
                        inc.append(df.columns[col])
                    else:
                        raise ValueError(
                            f"Invalid value for the stratify parameter. Value {col} "
                            f"is out of range for a dataset with {df.shape[1]} columns."
                        )
                elif isinstance(col, str):
                    if col in df:
                        inc.append(col)
                    else:
                        raise ValueError(
                            "Invalid value for the stratify parameter. "
                            f"Column {col} not found in the dataset."
                        )

            return df[inc]

    def _get_data(
        self,
        arrays: Sequence,
        y: Target = -1,
    ) -> tuple[DataFrame, DataContainer, DataFrame | None]:
        """Get data sets from a sequence of indexables.

        Also assigns an index, (stratified) shuffles and selects a
        subsample of rows depending on the attributes.

        Parameters
        ----------
        arrays: sequence of indexables
            Data set(s) provided. Should follow the API input format.

        y: int, str or sequence, default=-1
            Transformed target column.

        Returns
        -------
        dataframe
            Dataset containing the train and test sets.

        DataContainer
            Indices of the train and test sets.

        dataframe or None
            Holdout data set. Returns None if not specified.

        """

        def _subsample(df: DataFrame) -> DataFrame:
            """Select a random subset of a dataframe.

            If shuffle=True, the subset is shuffled, else row order
            is maintained. For forecasting tasks, rows are dropped
            from the tail of the data set.

            Parameters
            ----------
            df: dataframe
                Dataset.

            Returns
            -------
            dataframe
                Subset of df.

            """
            if self.n_rows <= 1:
                n_rows = int(len(df) * self.n_rows)
            else:
                n_rows = int(self.n_rows)

            if self.shuffle:
                return df.iloc[random.sample(range(len(df)), k=n_rows)]
            elif self.goal == "fc":
                return df.iloc[-n_rows:]  # For forecasting, select from tail
            else:
                return df.iloc[sorted(random.sample(range(len(df)), k=n_rows))]

        def _no_data_sets(
            X: DataFrame,
            y: Pandas,
        ) -> tuple[DataContainer, DataFrame | None]:
            """Generate data sets from one dataset.

            Additionally, assigns an index, shuffles the data, selects
            a subsample if `n_rows` is specified and split into sets in
            a stratified fashion.

            Parameters
            ----------
            X: dataframe
                Feature set with shape=(n_samples, n_features).

            y: series or dataframe
                Target column(s) corresponding to X.

            Returns
            -------
            DataContainer
                Train and test sets.

            dataframe or None
                Holdout data set. Returns None if not specified.

            """
            data = merge(X, y)

            # Shuffle the dataset
            if not 0 < self.n_rows <= len(data):
                raise ValueError(
                    "Invalid value for the n_rows parameter. Value should "
                    f"lie between 0 and len(X)={len(data)}, got {self.n_rows}."
                )
            data = _subsample(data)

            if isinstance(self._config.index, SequenceTypes):
                if len(self._config.index) != len(data):
                    raise IndexError(
                        "Invalid value for the index parameter. Length of "
                        f"index ({len(self._config.index)}) doesn't match "
                        f"that of the dataset ({len(data)})."
                    )
                data.index = self._config.index

            if len(data) < 5:
                raise ValueError(
                    f"The length of the dataset can't be <5, got {len(data)}. "
                    "Make sure n_rows=1 for small datasets."
                )

            if not 0 < self.test_size < len(data):
                raise ValueError(
                    "Invalid value for the test_size parameter. Value "
                    f"should lie between 0 and len(X), got {self.test_size}."
                )

            # Define test set size
            if self.test_size < 1:
                test_size = max(1, int(self.test_size * len(data)))
            else:
                test_size = self.test_size

            try:
                # Define holdout set size
                if self.holdout_size:
                    if self.holdout_size < 1:
                        holdout_size = max(1, int(self.holdout_size * len(data)))
                    else:
                        holdout_size = self.holdout_size

                    if not 0 <= holdout_size <= len(data) - test_size:
                        raise ValueError(
                            "Invalid value for the holdout_size parameter. "
                            "Value should lie between 0 and len(X) - len(test), "
                            f"got {self.holdout_size}."
                        )

                    data, holdout = train_test_split(
                        data,
                        test_size=holdout_size,
                        random_state=self.random_state,
                        shuffle=self._config.shuffle,
                        stratify=self._get_stratify_columns(data, y),
                    )
                else:
                    holdout = None

                train, test = train_test_split(
                    data,
                    test_size=test_size,
                    random_state=self.random_state,
                    shuffle=self._config.shuffle,
                    stratify=self._get_stratify_columns(data, y),
                )

                complete_set = self._set_index(bk.concat([train, test, holdout]), y)

                container = DataContainer(
                    data=(data := complete_set.iloc[:len(data)]),
                    train_idx=data.index[:-len(test)],
                    test_idx=data.index[-len(test):],
                    n_cols=len(get_cols(y)),
                )

            except ValueError as ex:
                # Clarify common error with stratification for multioutput tasks
                if "least populated class" in str(ex) and isinstance(y, DataFrameTypes):
                    raise ValueError(
                        "Stratification for multioutput tasks is applied over all target "
                        "columns, which results in a least populated class that has only "
                        "one member. Either select only one column to stratify over, or "
                        "set the parameter stratify=False."
                    )
                else:
                    raise ex

            if holdout is not None:
                holdout = complete_set.iloc[len(data):]

            return container, holdout

        def _has_data_sets(
            X_train: DataFrame,
            y_train: Pandas,
            X_test: DataFrame,
            y_test: Pandas,
            X_holdout: DataFrame | None = None,
            y_holdout: Pandas | None = None,
        ) -> tuple[DataContainer, DataFrame | None]:
            """Generate data sets from provided sets.

            Additionally, assigns an index, shuffles the data and
            selects a subsample if `n_rows` is specified.

            Parameters
            ----------
            X_train: dataframe
                Training set.

            y_train: series or dataframe
                Target column(s) corresponding to X_train.

            X_test: dataframe
                Test set.

            y_test: series or dataframe
                Target column(s) corresponding to X_test.

            X_holdout: dataframe or None
                Holdout set. Is None if not provided by the user.

            y_holdout: series, dataframe or None
                Target column(s) corresponding to X_holdout.

            Returns
            -------
            DataContainer
                Train and test sets.

            dataframe or None
                Holdout data set. Returns None if not specified.

            """
            train = merge(X_train, y_train)
            test = merge(X_test, y_test)
            if X_holdout is None:
                holdout = None
            else:
                holdout = merge(X_holdout, y_holdout)

            if not train.columns.equals(test.columns):
                raise ValueError("The train and test set do not have the same columns.")

            if holdout is not None:
                if not train.columns.equals(holdout.columns):
                    raise ValueError(
                        "The holdout set does not have the "
                        "same columns as the train and test set."
                    )

            # Skip the n_rows step if not called from atom
            # Don't use hasattr since getattr can fail when _models is not converted
            if "n_rows" in self.__dict__:
                if self.n_rows <= 1:
                    train = _subsample(train)
                    test = _subsample(test)
                    if holdout is not None:
                        holdout = _subsample(holdout)
                else:
                    raise ValueError(
                        "Invalid value for the n_rows parameter. Value must "
                        "be <1 when the train and test sets are provided."
                    )

            # If the index is a sequence, assign it before shuffling
            if isinstance(self._config.index, SequenceTypes):
                len_data = len(train) + len(test)
                if holdout is not None:
                    len_data += len(holdout)

                if len(self._config.index) != len_data:
                    raise IndexError(
                        "Invalid value for the index parameter. Length of "
                        f"index ({len(self._config.index)}) doesn't match "
                        f"that of the data sets ({len_data})."
                    )
                train.index = self._config.index[:len(train)]
                test.index = self._config.index[len(train):len(train) + len(test)]
                if holdout is not None:
                    holdout.index = self._config.index[-len(holdout):]

            complete_set = self._set_index(bk.concat([train, test, holdout]), y)

            container = DataContainer(
                data=(data := complete_set.iloc[:len(train) + len(test)]),
                train_idx=data.index[:len(train)],
                test_idx=data.index[-len(test):],
                n_cols=len(get_cols(y_train)),
            )

            if holdout is not None:
                holdout = complete_set.iloc[len(train) + len(test):]

            return container, holdout

        # Process input arrays ===================================== >>

        if len(arrays) == 0:
            if self.goal == "fc" and not isinstance(y, (Int, str)):
                # arrays=() and y=y for forecasting
                sets = _no_data_sets(*self._prepare_input(y=y))
            elif not self.branch._data:
                raise ValueError(
                    "The data arrays are empty! Provide the data to run the pipeline "
                    "successfully. See the documentation for the allowed formats."
                )
            else:
                return self.branch._data, self.branch._holdout

        elif len(arrays) == 1:
            # arrays=(X,) or arrays=(y,) for forecasting
            sets = _no_data_sets(*self._prepare_input(arrays[0], y=y))

        elif len(arrays) == 2:
            if isinstance(arrays[0], tuple) and len(arrays[0]) == len(arrays[1]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test))
                X_train, y_train = self._prepare_input(arrays[0][0], arrays[0][1])
                X_test, y_test = self._prepare_input(arrays[1][0], arrays[1][1])
                sets = _has_data_sets(X_train, y_train, X_test, y_test)
            elif isinstance(arrays[1], (*IntTypes, str)) or n_cols(arrays[1]) == 1:
                try:
                    # arrays=(X, y)
                    sets = _no_data_sets(*self._prepare_input(arrays[0], arrays[1]))
                except ValueError as ex:
                    if self.goal == "fc":
                        # arrays=(train, test) for forecast
                        X_train, y_train = self._prepare_input(y=arrays[0])
                        X_test, y_test = self._prepare_input(y=arrays[1])
                        sets = _has_data_sets(X_train, y_train, X_test, y_test)
                    else:
                        raise ex
            else:
                # arrays=(train, test)
                X_train, y_train = self._prepare_input(arrays[0], y=y)
                X_test, y_test = self._prepare_input(arrays[1], y=y)
                sets = _has_data_sets(X_train, y_train, X_test, y_test)

        elif len(arrays) == 3:
            if len(arrays[0]) == len(arrays[1]) == len(arrays[2]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test), (X_holdout, y_holdout))
                X_train, y_train = self._prepare_input(arrays[0][0], arrays[0][1])
                X_test, y_test = self._prepare_input(arrays[1][0], arrays[1][1])
                X_hold, y_hold = self._prepare_input(arrays[2][0], arrays[2][1])
                sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)
            else:
                # arrays=(train, test, holdout)
                X_train, y_train = self._prepare_input(arrays[0], y=y)
                X_test, y_test = self._prepare_input(arrays[1], y=y)
                X_hold, y_hold = self._prepare_input(arrays[2], y=y)
                sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)

        elif len(arrays) == 4:
            # arrays=(X_train, X_test, y_train, y_test)
            X_train, y_train = self._prepare_input(arrays[0], arrays[2])
            X_test, y_test = self._prepare_input(arrays[1], arrays[3])
            sets = _has_data_sets(X_train, y_train, X_test, y_test)

        elif len(arrays) == 6:
            # arrays=(X_train, X_test, X_holdout, y_train, y_test, y_holdout)
            X_train, y_train = self._prepare_input(arrays[0], arrays[3])
            X_test, y_test = self._prepare_input(arrays[1], arrays[4])
            X_hold, y_hold = self._prepare_input(arrays[2], arrays[5])
            sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)

        else:
            raise ValueError(
                "Invalid data arrays. See the documentation for the allowed formats."
            )

        if self.goal == "fc":
            # For forecasting, check if index complies with sktime's standard
            valid, msg, _ = check_is_mtype(
                obj=pd.DataFrame(bk.concat([sets[0].data, sets[1]])),
                mtype="pd.DataFrame",
                return_metadata=True,
                var_name="the dataset",
            )

            if not valid:
                raise ValueError(msg)
        else:
            # Else check for duplicate indices
            if bk.concat([sets[0].data, sets[1]]).index.duplicated().any():
                raise ValueError(
                    "Duplicate indices found in the dataset. "
                    "Try initializing atom using `index=False`."
                )

        return sets

    @crash
    def _log(self, msg: Scalar | str, level: Int = 0, severity: Severity = "info"):
        """Print message and save to log file.

        Parameters
        ----------
        msg: int, float or str
            Message to save to the logger and print to stdout.

        level: int, default=0
            Minimum verbosity level to print the message.

        severity: str, default="info"
            Severity level of the message. Choose from: debug, info,
            warning, error, critical.

        """
        if severity in ("error", "critical"):
            raise UserWarning(msg)
        elif severity == "warning":
            warnings.warn(msg, category=UserWarning)
        elif severity == "info" and self.verbose >= level:
            print(msg)

        if self.logger:
            for text in str(msg).split("\n"):
                getattr(self.logger, severity)(str(text))
