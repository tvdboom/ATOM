# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseTransformer class.

"""

from __future__ import annotations

import multiprocessing
import os
import random
import sys
import warnings
from copy import deepcopy
from datetime import datetime as dt
from importlib import import_module
from importlib.util import find_spec
from logging import DEBUG, FileHandler, Formatter, Logger, getLogger

import dill as pickle
import mlflow
import numpy as np
import ray
import sklearnex
from ray.util.joblib import register_ray
from sklearn.model_selection import train_test_split
from typeguard import typechecked

from atom.utils import (
    DATAFRAME, DATAFRAME_TYPES, FEATURES, INDEX, INT, INT_TYPES, PANDAS,
    SCALAR, SEQUENCE, SEQUENCE_TYPES, TARGET, Predictor, bk, composed, crash,
    get_cols, lst, merge, method_to_log, n_cols, pd, to_df, to_pandas,
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
        - device: Device on which to train the estimators.
        - engine: Execution engine to use for the estimators.
        - backend: Parallelization backend.
        - verbose: Verbosity level of the output.
        - warnings: Whether to show or suppress encountered warnings.
        - logger: Name of the log file or Logger object.
        - experiment: Name of the mlflow experiment used for tracking.
        - random_state: Seed used by the random number generator.

    """

    attrs = [
        "n_jobs",
        "device",
        "engine",
        "backend",
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
    def n_jobs(self) -> INT:
        """Number of cores to use for parallel processing."""
        return self._n_jobs

    @n_jobs.setter
    @typechecked
    def n_jobs(self, value: INT):
        # Check number of cores for multiprocessing
        n_cores = multiprocessing.cpu_count()
        if value > n_cores:
            value = n_cores
        else:
            value = n_cores + 1 + value if value < 0 else value

            # Final check for negative input
            if value < 1:
                raise ValueError(
                    f"Invalid value for the n_jobs parameter, got {value}.", 1
                )

        self._n_jobs = value

    @property
    def device(self) -> str:
        """Device on which to train the estimators."""
        return self._device

    @device.setter
    @typechecked
    def device(self, value: str):
        self._device = value
        if "gpu" in value.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._device_id)

    @property
    def engine(self) -> str:
        """Execution engine for estimators."""
        return self._engine

    @engine.setter
    @typechecked
    def engine(self, value: str):
        if value == "sklearnex":
            sklearnex.set_config("auto" if "cpu" in self.device else self.device)
        elif value == "cuml":
            if "cpu" in self.device:
                raise ValueError(
                    f"Invalid value for the engine parameter. device="
                    f"{self.device} only supports sklearn or sklearnex."
                )
            elif not find_spec("cuml"):
                raise ModuleNotFoundError(
                    "Failed to import cuml. Package is not installed. Refer "
                    "to: https://rapids.ai/start.html#rapids-release-selector."
                )
        elif value != "sklearn":
            raise ValueError(
                "Invalid value for the engine parameter, got "
                f"{value}. Choose from: sklearn, sklearnex, cuml."
            )

        self._engine = value

    @property
    def backend(self) -> str:
        """Parallelization backend."""
        return self._backend

    @backend.setter
    @typechecked
    def backend(self, value: str):
        options = ("loky", "multiprocessing", "threading", "ray")
        if value == "ray":
            import modin.pandas as md

            # Overwrite utils backend with modin
            for module in sys.modules:
                if module.startswith("atom."):
                    setattr(sys.modules[module], "bk", md)

            register_ray()  # Register ray as joblib backend
            if not ray.is_initialized():
                ray.init(
                    runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}},
                    log_to_driver=False,
                )

        elif value not in options:
            raise ValueError(
                f"Invalid value for the backend parameter, got "
                f"{value}. Choose from: {', '.join(options)}."
            )
        else:
            # Overwrite utils backend with pandas
            for module in sys.modules:
                if module.startswith("atom."):
                    setattr(sys.modules[module], "bk", pd)

        self._backend = value

    @property
    def verbose(self) -> INT:
        """Verbosity level of the output."""
        return self._verbose

    @verbose.setter
    @typechecked
    def verbose(self, value: INT):
        if value < 0 or value > 2:
            raise ValueError(
                "Invalid value for the verbose parameter. Value"
                f" should be between 0 and 2, got {value}."
            )
        self._verbose = value

    @property
    def warnings(self) -> str:
        """Whether to show or suppress encountered warnings."""
        return self._warnings

    @warnings.setter
    @typechecked
    def warnings(self, value: bool | str):
        if isinstance(value, bool):
            self._warnings = "default" if value else "ignore"
        else:
            options = ("default", "error", "ignore", "always", "module", "once")
            if value not in options:
                raise ValueError(
                    "Invalid value for the warnings parameter, got "
                    f"{value}. Choose from: {', '.join(options)}."
                )
            self._warnings = value

        warnings.filterwarnings(self._warnings)  # Change the filter in this process
        warnings.filterwarnings("ignore", category=UserWarning, module=".*modin.*")
        os.environ["PYTHONWARNINGS"] = self._warnings  # Affects subprocesses (joblib)

    @property
    def logger(self) -> Logger:
        """Name of the log file or Logger object."""
        return self._logger

    @logger.setter
    @typechecked
    def logger(self, value: str | Logger | None):
        # Loggers from external libraries to redirect to the file handler
        external_loggers = ["mlflow", "optuna", "ray", "modin", "featuretools", "gradio"]

        if not value:
            self._logger = None
            for logger in external_loggers:
                getLogger(logger).handlers.clear()

        elif isinstance(value, str):
            # Prepare the FileHandler
            if not value.endswith(".log"):
                value += ".log"
            if value.endswith("auto.log"):
                current = dt.now().strftime("%d%b%y_%Hh%Mm%Ss")
                value = value.replace("auto", self.__class__.__name__ + "_" + current)

            handler = FileHandler(value)
            handler.setFormatter(Formatter("%(asctime)s - %(levelname)s: %(message)s"))

            self._logger = getLogger(self.__class__.__name__)
            self._logger.setLevel(DEBUG)

            # Redirect loggers to file handler
            for logger in [self._logger.name] + external_loggers:
                for h in (log := getLogger(logger)).handlers:
                    h.close()  # Close existing handlers
                log.handlers.clear()
                log.addHandler(handler)

        else:
            self._logger = value

    @property
    def experiment(self) -> str | None:
        """Name of the mlflow experiment used for tracking."""
        return self._experiment

    @experiment.setter
    @typechecked
    def experiment(self, value: str | None):
        self._experiment = value
        if value:
            mlflow.sklearn.autolog(disable=True)
            mlflow.set_experiment(value)

    @property
    def random_state(self) -> INT:
        """Seed used by the random number generator."""
        return self._random_state

    @random_state.setter
    @typechecked
    def random_state(self, value: INT | None):
        if value and value < 0:
            raise ValueError(
                "Invalid value for the random_state parameter. "
                f"Value should be >0, got {value}."
            )
        random.seed(value)
        np.random.seed(value)
        self._random_state = value

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

    def _get_est_class(self, name: str, module: str) -> Predictor:
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
        Predictor
            Class of the estimator.

        """
        try:
            return getattr(import_module(f"{self.engine}.{module}"), name)
        except (ModuleNotFoundError, AttributeError):
            return getattr(import_module(f"sklearn.{module}"), name)

    @staticmethod
    @typechecked
    def _prepare_input(
        X: FEATURES | None = None,
        y: TARGET | None = None,
    ) -> tuple[DATAFRAME | None, PANDAS | None]:
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
            - If sequence: Target array with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        dataframe or None
            Feature dataset. Only returned if provided.

        series, dataframe or None
            Target column(s) corresponding to X.

        """
        if X is None and y is None:
            raise ValueError("X and y can't be both None!")
        elif X is not None:
            X = to_df(deepcopy(X))  # Make copy to not overwrite mutable arguments

            # If text dataset, change the name of the column to corpus
            if list(X.columns) == ["x0"] and X[X.columns[0]].dtype == "object":
                X = X.rename(columns={X.columns[0]: "corpus"})

            # Convert all column names to str
            X.columns = map(str, X.columns)

            # No duplicate column names are allowed
            if len(set(X.columns)) != len(X.columns):
                raise ValueError("Duplicate column names found in X.")

        # Prepare target column
        if isinstance(y, (dict, *SEQUENCE_TYPES, DATAFRAME_TYPES)):
            if isinstance(y, dict):
                if n_cols(y := to_df(y, index=getattr(X, "index", None))) == 1:
                    y = y.iloc[:, 0]  # If y is one-dimensional, get series

            elif isinstance(y, (SEQUENCE_TYPES, DATAFRAME_TYPES)):
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
                    data=y,
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

        elif isinstance(y, INT_TYPES):
            if X is None:
                raise ValueError("X can't be None when y is an int.")

            X, y = X.drop(X.columns[y], axis=1), X[X.columns[y]]

        return X, y

    def _set_index(self, df: DATAFRAME, y: PANDAS | None) -> DATAFRAME:
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
        if self.index is True:  # True gets caught by isinstance(int)
            return df
        elif self.index is False:
            df = df.reset_index(drop=True)
        elif isinstance(self.index, INT_TYPES):
            if -df.shape[1] <= self.index <= df.shape[1]:
                df = df.set_index(df.columns[self.index], drop=True)
            else:
                raise ValueError(
                    f"Invalid value for the index parameter. Value {self.index} "
                    f"is out of range for a dataset with {df.shape[1]} columns."
                )
        elif isinstance(self.index, str):
            if self.index in df:
                df = df.set_index(self.index, drop=True)
            else:
                raise ValueError(
                    "Invalid value for the index parameter. "
                    f"Column {self.index} not found in the dataset."
                )

        if y is not None and df.index.name in (c.name for c in get_cols(y)):
            raise ValueError(
                "Invalid value for the index parameter. The index column "
                f"can not be the same as the target column, got {df.index.name}."
            )

        return df

    def _get_stratify_columns(self, df: DATAFRAME) -> DATAFRAME | None:
        """Get columns to stratify by.

        Parameters
        ----------
        df: dataframe
            Dataset from which to get the columns.

        Returns
        -------
        dataframe or None
            Dataset with subselection of columns. Returns None if
            there's no stratification.

        """
        # Stratification is not possible when the data cannot change order
        if self.stratify is False:
            return None
        elif self.shuffle is False:
            self.log(
                "Stratification is not possible when shuffle=False.", 3, "warning"
            )
            return None
        elif self.stratify is True:
            return df.iloc[:, -1]
        else:
            inc = []
            for col in lst(self.stratify):
                if isinstance(col, INT_TYPES):
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
        arrays: SEQUENCE,
        y: TARGET = -1,
        use_n_rows: bool = True,
    ) -> tuple[DATAFRAME, list[INDEX], DATAFRAME | None]:
        """Get data sets from a sequence of indexables.

        Also assigns an index, (stratified) shuffles and selects a
        subsample of rows depending on the attributes.

        Parameters
        ----------
        arrays: sequence of indexables
            Data set(s) provided. Should follow the API input format.

        y: int, str or sequence, default=-1
            Transformed target column.

        use_n_rows: bool, default=True
            Whether to use the `n_rows` parameter on the dataset.

        Returns
        -------
        dataframe
            Dataset containing the train and test sets.

        list or Index
            Indices of the train and test sets.

        dataframe or None
            Holdout data set. Returns None if not specified.

        """

        def _subsample(df: DATAFRAME) -> DATAFRAME:
            """Select a random subset of a dataframe.

            If shuffle=True, the subset is shuffled, else row order
            is maintained.

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
            else:
                return df.iloc[sorted(random.sample(range(len(df)), k=n_rows))]

        def _no_data_sets(
            X: DATAFRAME,
            y: PANDAS,
        ) -> tuple[DATAFRAME, list, DATAFRAME | None]:
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
            tuple
                Data, indices and holdout.

            """
            data = merge(X, y)

            # If the index is a sequence, assign it before shuffling
            if isinstance(self.index, SEQUENCE_TYPES):
                if len(self.index) != len(data):
                    raise ValueError(
                        "Invalid value for the index parameter. Length of "
                        f"index ({len(self.index)}) doesn't match that of "
                        f"the dataset ({len(data)})."
                    )
                data.index = self.index

            if use_n_rows:
                if not 0 < self.n_rows <= len(data):
                    raise ValueError(
                        "Invalid value for the n_rows parameter. Value "
                        f"should lie between 0 and len(X), got {self.n_rows}."
                    )
                data = _subsample(data)

            if len(data) < 5:
                raise ValueError(
                    "Invalid value for the n_rows parameter. The "
                    f"length of the dataset can't be <5, got {self.n_rows}."
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
                    shuffle=self.shuffle,
                    stratify=self._get_stratify_columns(data),
                )
                holdout = self._set_index(holdout, y)
            else:
                holdout = None

            train, test = train_test_split(
                data,
                test_size=test_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=self._get_stratify_columns(data),
            )
            data = self._set_index(bk.concat([train, test]), y)

            # [number of target columns, train indices, test indices]
            idx = [len(get_cols(y)), data.index[:-test_size], data.index[-test_size:]]

            return data, idx, holdout

        def _has_data_sets(
            X_train: DATAFRAME,
            y_train: PANDAS,
            X_test: DATAFRAME,
            y_test: PANDAS,
            X_holdout: DATAFRAME | None = None,
            y_holdout: PANDAS | None = None,
        ) -> tuple[DATAFRAME, list, DATAFRAME | None]:
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
            tuple
                Data, indices and holdout.

            """
            train = merge(X_train, y_train)
            test = merge(X_test, y_test)
            holdout = merge(X_holdout, y_holdout) if X_holdout is not None else None

            # If the index is a sequence, assign it before shuffling
            if isinstance(self.index, SEQUENCE_TYPES):
                len_data = len(train) + len(test)
                if holdout is not None:
                    len_data += len(holdout)

                if len(self.index) != len_data:
                    raise ValueError(
                        "Invalid value for the index parameter. Length of "
                        f"index ({len(self.index)}) doesn't match that of "
                        f"the data sets ({len_data})."
                    )
                train.index = self.index[:len(train)]
                test.index = self.index[len(train):len(train) + len(test)]
                if holdout is not None:
                    holdout.index = self.index[-len(holdout):]

            # Skip the n_rows step if not called from atom
            # Don't use hasattr since getattr can fail when _models is not converted
            if "n_rows" in self.__dict__ and use_n_rows:
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

            if not train.columns.equals(test.columns):
                raise ValueError("The train and test set do not have the same columns.")

            if holdout is not None:
                if not train.columns.equals(holdout.columns):
                    raise ValueError(
                        "The holdout set does not have the "
                        "same columns as the train and test set."
                    )
                holdout = self._set_index(holdout, y_train)

            data = self._set_index(bk.concat([train, test]), y_train)

            # [number of target columns, train indices, test indices]
            idx = [
                len(get_cols(y_train)),
                data.index[:len(train)],
                data.index[-len(test):],
            ]

            return data, idx, holdout

        # Process input arrays ===================================== >>

        if len(arrays) == 0:
            if self.branch._data is None:
                raise ValueError(
                    "The data arrays are empty! Provide the data to run the pipeline "
                    "successfully. See the documentation for the allowed formats."
                )
            else:
                return self.branch._data, self.branch._idx, self.branch._holdout

        elif len(arrays) == 1:
            # arrays=(X,)
            sets = _no_data_sets(*self._prepare_input(arrays[0], y=y))

        elif len(arrays) == 2:
            if isinstance(arrays[0], tuple) and len(arrays[0]) == len(arrays[1]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test))
                X_train, y_train = self._prepare_input(arrays[0][0], arrays[0][1])
                X_test, y_test = self._prepare_input(arrays[1][0], arrays[1][1])
                sets = _has_data_sets(X_train, y_train, X_test, y_test)
            elif isinstance(arrays[1], (int, str)) or n_cols(arrays[1]) == 1:
                # arrays=(X, y)
                sets = _no_data_sets(*self._prepare_input(arrays[0], arrays[1]))
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

        return sets

    @composed(crash, typechecked)
    def log(self, msg: SCALAR | str, level: INT = 0, severity: str = "info"):
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
        if severity not in ("debug", "info", "warning", "error", "critical"):
            raise ValueError(
                "Invalid value for the severity parameter. Choose "
                "from: debug, info, warning, error, critical."
            )

        if severity in ("error", "critical"):
            raise UserWarning(msg)
        elif severity == "warning":
            warnings.warn(msg)
        elif severity == "info" and self.verbose >= level:
            print(msg)

        # Store in file
        if self.logger is not None:
            for text in str(msg).split("\n"):
                getattr(self.logger, severity)(str(text))

    @composed(crash, method_to_log, typechecked)
    def save(self, filename: str = "auto", *, save_data: bool = True):
        """Save the instance to a pickle file.

        Parameters
        ----------
        filename: str, default="auto"
            Name of the file. Use "auto" for automatic naming.

        save_data: bool, default=True
            Whether to save the dataset with the instance. This parameter
            is ignored if the method is not called from atom. If False,
            add the data to the [load][atomclassifier-load] method.

        """
        if not save_data and hasattr(self, "dataset"):
            data = {}
            for branch in self._branches:
                data[branch.name] = dict(
                    data=deepcopy(branch._data),
                    holdout=deepcopy(branch._holdout),
                )
                branch._data = None
                branch._holdout = None
                branch.__dict__.pop("holdout", None)  # Clear cached holdout

        if filename.endswith("auto"):
            filename = filename.replace("auto", self.__class__.__name__)

        with open(filename, "wb") as f:
            pickle.settings["recurse"] = True
            pickle.dump(self, f)

        # Restore the data to the attributes
        if not save_data and hasattr(self, "dataset"):
            for branch in self._branches:
                branch._data = data[branch.name]["data"]
                branch._holdout = data[branch.name]["holdout"]

        self.log(f"{self.__class__.__name__} successfully saved.", 1)
