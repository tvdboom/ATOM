# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseTransformer class.

"""

import multiprocessing
import os
import random
import warnings
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from logging import Logger
from typing import List, Optional, Tuple, Union

import dill as pickle
import mlflow
import numpy as np
import pandas as pd
import sklearnex
from sklearn.model_selection import train_test_split
from typeguard import typechecked

from atom.utils import (
    INT, SCALAR, SEQUENCE, SEQUENCE_TYPES, X_TYPES, Y_TYPES, Estimator,
    composed, crash, lst, merge, method_to_log, prepare_logger, to_df,
    to_series,
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
        """Execution engine to use for the estimators."""
        return self._engine

    @engine.setter
    @typechecked
    def engine(self, value: str):
        if value.lower() == "sklearnex":
            sklearnex.set_config(target_offload=self.device)
        elif value.lower() == "cuml":
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
        elif value.lower() != "sklearn":
            raise ValueError(
                "Invalid value for the engine parameter, got "
                f"{value}. Choose from : sklearn, sklearnex, cuml."
            )

        self._engine = value

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
    def warnings(self, value: Union[bool, str]):
        if isinstance(value, bool):
            self._warnings = "default" if value else "ignore"
        else:
            opts = ["error", "ignore", "always", "default", "module", "once"]
            if value.lower() not in opts:
                raise ValueError(
                    "Invalid value for the warnings parameter, got "
                    f"{value}. Choose from: {', '.join(opts)}."
                )
            self._warnings = value.lower()

        warnings.simplefilter(self._warnings)  # Change the filter in this process
        os.environ["PYTHONWARNINGS"] = self._warnings  # Affects subprocesses

    @property
    def logger(self) -> Logger:
        """Name of the log file or Logger object."""
        return self._logger

    @logger.setter
    @typechecked
    def logger(self, value: Optional[Union[str, Logger]]):
        self._logger = prepare_logger(value, class_name=self.__class__.__name__)

    @property
    def experiment(self) -> Optional[str]:
        """Name of the mlflow experiment used for tracking."""
        return self._experiment

    @experiment.setter
    @typechecked
    def experiment(self, value: Optional[str]):
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
    def random_state(self, value: Optional[INT]):
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
            return getattr(import_module(f"{self.engine}.{module}"), name)
        except (ModuleNotFoundError, AttributeError):
            return getattr(import_module(f"sklearn.{module}"), name)

    @staticmethod
    @typechecked
    def _prepare_input(
        X: Optional[X_TYPES] = None,
        /,
        y: Optional[Y_TYPES] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare the input data.

        Convert X and y to pandas (if not already) and perform standard
        compatibility checks (dimensions, length, indices, etc...).

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.
                - If None: y is ignored.
                - If int: Position of the target column in X.
                - If str: Name of the target column in X.
                - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        pd.DataFrame or None
            Feature dataset. Only returned if provided.

        pd.Series or None
            Transformed target column. Only returned if provided.

        """
        if X is None and y is None:
            raise ValueError("X and y can't be both None!")
        elif X is not None:
            X = to_df(deepcopy(X))  # Make copy to not overwrite mutable arguments

            # If text dataset, change the name of the column to corpus
            if list(X.columns) == ["x0"] and X[X.columns[0]].dtype == "object":
                X = X.rename(columns={X.columns[0]: "corpus"})

            # Convert all column names to str
            X.columns = [str(col) for col in X.columns]

        # Prepare target column
        if isinstance(y, (dict, *SEQUENCE)):
            if not isinstance(y, pd.Series):
                # Check that y is one-dimensional
                if ndim := np.array(y).ndim > 1:
                    raise ValueError(f"y should be one-dimensional, got ndim={ndim}.")

                # Check X and y have the same number of rows
                if X is not None and len(X) != len(y):
                    raise ValueError(
                        "X and y don't have the same number of rows,"
                        f" got len(X)={len(X)} and len(y)={len(y)}."
                    )

                y = to_series(y, index=getattr(X, "index", None))

            elif X is not None and not X.index.equals(y.index):
                raise ValueError("X and y don't have the same indices!")

        elif isinstance(y, str):
            if X is not None:
                if y not in X.columns:
                    raise ValueError(f"Column {y} not found in X!")

                X, y = X.drop(y, axis=1), X[y]

            else:
                raise ValueError("X can't be None when y is a string.")

        elif isinstance(y, int):
            if X is None:
                raise ValueError("X can't be None when y is an int.")

            X, y = X.drop(X.columns[y], axis=1), X[X.columns[y]]

        return X, y

    def _set_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign an index to the dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            Dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with updated indices.

        """
        target = df.columns[-1]

        if self.index is True:  # True gets caught by isinstance(int)
            return df
        elif self.index is False:
            df = df.reset_index(drop=True)
        elif isinstance(self.index, int):
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

        if df.index.name == target:
            raise ValueError(
                "Invalid value for the index parameter. The index column "
                f"can not be the same as the target column, got {target}."
            )

        return df

    def _get_stratify_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get columns to stratify by.

        Parameters
        ----------
        df: pd.DataFrame
            Dataset from which to get the columns.

        Returns
        -------
        pd.DataFrame or None
            Dataset with subselection of columns. Returns None if
            there's no stratification.

        """
        # Stratification is not possible when the data can not change order
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
                if isinstance(col, int):
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
        arrays: SEQUENCE_TYPES,
        y: Y_TYPES = -1,
        use_n_rows: bool = True,
    ) -> Tuple[pd.DataFrame, List[pd.Index], Optional[pd.DataFrame]]:
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
        pd.DataFrame
            Dataset containing the train and test sets.

        list of pd.Index
            Indices of the train and test sets.

        pd.DataFrame or None
            Holdout data set. Returns None if not specified.

        """

        def _no_data_sets(data: pd.DataFrame) -> tuple:
            """Generate data sets from one dataframe.

            Additionally, assigns an index, shuffles the data, selects
            a subsample if `n_rows` is specified and split into sets in
            a stratified fashion.

            Parameters
            ----------
            data: pd.DataFrame
                Complete dataset containing all data sets.

            Returns
            -------
            tuple
                Data, indices and holdout.

            """
            # If the index is a sequence, assign it before shuffling
            if isinstance(self.index, SEQUENCE):
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

                # Select subset of the data
                if self.n_rows > 1:
                    data = data.iloc[:int(self.n_rows), :]
                else:
                    data = data.iloc[:int(len(data) * self.n_rows), :]

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
                holdout = self._set_index(holdout)
            else:
                holdout = None

            train, test = train_test_split(
                data,
                test_size=test_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=self._get_stratify_columns(data),
            )
            data = self._set_index(pd.concat([train, test]))

            return data, [data.index[:-test_size], data.index[-test_size:]], holdout

        def _has_data_sets(
            train: pd.DataFrame,
            test: pd.DataFrame,
            holdout: Optional[pd.DataFrame] = None,
        ) -> tuple:
            """Generate data sets from provided sets.

            Additionally, assigns an index, shuffles the data and
            selects a subsample if `n_rows` is specified.

            Parameters
            ----------
            train: pd.DataFrame
                Training set.

            test: pd.DataFrame
                Test set.

            holdout: pd.DataFrame or None
                Holdout set. Is None if not provided by the user.

            Returns
            -------
            tuple
                Data, indices and holdout.

            """
            # If the index is a sequence, assign it before shuffling
            if isinstance(self.index, SEQUENCE):
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
            if hasattr(self, "n_rows") and use_n_rows:
                # Select same subsample of train and test set
                if self.n_rows <= 1:
                    n_train = int(len(train) * self.n_rows)
                    n_test = int(len(test) * self.n_rows)
                    if self.shuffle:
                        train = train.sample(n=n_train, random_state=self.random_state)
                        test = test.sample(n=n_test, random_state=self.random_state)
                    else:
                        train = train.iloc[:n_train, :]
                        test = test.iloc[:n_test, :]

                    if holdout is not None:
                        n_holdout = int(len(holdout) * self.n_rows)
                        if self.shuffle:
                            holdout = holdout.sample(
                                n=n_holdout,
                                random_state=self.random_state,
                            )
                        else:
                            holdout = holdout.iloc[:n_holdout, :]

                else:
                    raise ValueError(
                        "Invalid value for the n_rows parameter. Value has "
                        "to be <1 when a train and test set are provided."
                    )

            if not train.columns.equals(test.columns):
                raise ValueError("The train and test set do not have the same columns.")

            if holdout is not None:
                if not train.columns.equals(holdout.columns):
                    raise ValueError(
                        "The holdout set does not have the "
                        "same columns as the train and test set."
                    )
                holdout = self._set_index(holdout)

            data = self._set_index(pd.concat([train, test]))
            idx = [data.index[:len(train)], data.index[-len(test):]]

            return data, idx, holdout

        # Process input arrays ===================================== >>

        if len(arrays) == 0:
            if self.branch._data is None:
                raise ValueError(
                    "The data arrays are empty! Provide the data to run the pipeline "
                    "successfully. See the documentation for the allowed formats."
                )
            else:
                return self.branch._data, self.branch._idx, self.holdout

        elif len(arrays) == 1:
            # arrays=(X,)
            data = merge(*self._prepare_input(arrays[0], y=y))
            sets = _no_data_sets(data)

        elif len(arrays) == 2:
            if isinstance(arrays[0], tuple) and len(arrays[0]) == len(arrays[1]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test))
                train = merge(*self._prepare_input(arrays[0][0], arrays[0][1]))
                test = merge(*self._prepare_input(arrays[1][0], arrays[1][1]))
                sets = _has_data_sets(train, test)
            elif isinstance(arrays[1], (int, str)) or np.array(arrays[1]).ndim == 1:
                # arrays=(X, y)
                data = merge(*self._prepare_input(arrays[0], arrays[1]))
                sets = _no_data_sets(data)
            else:
                # arrays=(train, test)
                train = merge(*self._prepare_input(arrays[0], y=y))
                test = merge(*self._prepare_input(arrays[1], y=y))
                sets = _has_data_sets(train, test)

        elif len(arrays) == 3:
            if len(arrays[0]) == len(arrays[1]) == len(arrays[2]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test), (X_holdout, y_holdout))
                train = merge(*self._prepare_input(arrays[0][0], arrays[0][1]))
                test = merge(*self._prepare_input(arrays[1][0], arrays[1][1]))
                holdout = merge(*self._prepare_input(arrays[2][0], arrays[2][1]))
                sets = _has_data_sets(train, test, holdout)
            else:
                # arrays=(train, test, holdout)
                train = merge(*self._prepare_input(arrays[0], y=y))
                test = merge(*self._prepare_input(arrays[1], y=y))
                holdout = merge(*self._prepare_input(arrays[2], y=y))
                sets = _has_data_sets(train, test, holdout)

        elif len(arrays) == 4:
            # arrays=(X_train, X_test, y_train, y_test)
            train = merge(*self._prepare_input(arrays[0], arrays[2]))
            test = merge(*self._prepare_input(arrays[1], arrays[3]))
            sets = _has_data_sets(train, test)

        elif len(arrays) == 6:
            # arrays=(X_train, X_test, X_holdout, y_train, y_test, y_holdout)
            train = merge(*self._prepare_input(arrays[0], arrays[3]))
            test = merge(*self._prepare_input(arrays[1], arrays[4]))
            holdout = merge(*self._prepare_input(arrays[2], arrays[5]))
            sets = _has_data_sets(train, test, holdout)

        else:
            raise ValueError(
                "Invalid data arrays. See the documentation for the allowed formats."
            )

        return sets

    @composed(crash, typechecked)
    def log(self, msg: Union[SCALAR, str], level: INT = 0, severity: str = "info"):
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

        if self.verbose >= level:
            print(msg)

        if severity == "warning":
            warnings.warn(msg)

        if self.logger is not None:
            for text in str(msg).split("\n"):
                getattr(self.logger, severity)(str(text))

    @composed(crash, method_to_log, typechecked)
    def save(self, filename: str = "auto", save_data: bool = True):
        """Save the instance to a pickle file.

        Parameters
        ----------
        filename: str, default="auto"
            Name of the file. Use "auto" for automatic naming.

        save_data: bool, default=True
            Whether to save the dataset with the instance. This
            parameter is ignored if the method is not called from
            atom. If False, remember to add the data to [ATOMLoader][]
            when loading the file.

        """
        if not save_data and hasattr(self, "dataset"):
            data = {"holdout": deepcopy(self.holdout)}  # Store data to reattach later
            self.holdout = None
            for key, value in self._branches.items():
                data[key] = deepcopy(value._data)
                value._data = None

        if filename.endswith("auto"):
            filename = filename.replace("auto", self.__class__.__name__)

        with open(filename, "wb") as f:
            pickle.settings["recurse"] = True
            pickle.dump(self, f)  # Dill replaces pickle to dump lambdas

        # Restore the data to the attributes
        if not save_data and hasattr(self, "dataset"):
            self.holdout = data["holdout"]
            for key, value in self._branches.items():
                value._data = data[key]

        self.log(f"{self.__class__.__name__} successfully saved.", 1)
