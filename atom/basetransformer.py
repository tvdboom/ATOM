# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: Mavs
Description: Module containing the BaseTransformer class.

"""

# Standard packages
import os
import dill
import random
import mlflow
import warnings
import numpy as np
import pandas as pd
import multiprocessing
from copy import deepcopy
from typeguard import typechecked
from typing import Union, Optional

# Own modules
from .utils import (
    SEQUENCE, X_TYPES, Y_TYPES, to_df, to_series, merge,
    prepare_logger, composed, crash, method_to_log,
)


class BaseTransformer:
    """Base class for estimators in the package.

    Contains shared properties (n_jobs, verbose, warnings, logger,
    random_state) and standard methods across estimators.

    Parameters
    ----------
    **kwargs
        Standard keyword arguments for the classes. Can include:
            - n_jobs: Number of cores to use for parallel processing.
            - verbose: Verbosity level of the output.
            - warnings: Whether to show or suppress encountered warnings.
            - logger: Name of the log file or Logger object.
            - experiment: Name of the mlflow experiment used for tracking.
            - random_state: Seed used by the random number generator.

    """

    attrs = ["n_jobs", "verbose", "warnings", "logger", "experiment", "random_state"]

    def __init__(self, **kwargs):
        """Update the properties with the provided kwargs."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Properties =================================================== >>

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    @typechecked
    def n_jobs(self, value: int):
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
    def verbose(self):
        return self._verbose

    @verbose.setter
    @typechecked
    def verbose(self, value: int):
        if value < 0 or value > 2:
            raise ValueError(
                "Invalid value for the verbose parameter. Value"
                f" should be between 0 and 2, got {value}."
            )
        self._verbose = value

    @property
    def warnings(self):
        return self._warnings

    @warnings.setter
    @typechecked
    def warnings(self, value: Union[bool, str]):
        if isinstance(value, bool):
            self._warnings = "default" if value else "ignore"
        else:
            opts = ["error", "ignore", "always", "default", "module", "once"]
            if value not in opts:
                raise ValueError(
                    "Invalid value for the warnings parameter, got "
                    f"{value}. Choose from: {', '.join(opts)}."
                )
            self._warnings = value

        warnings.simplefilter(self._warnings)  # Change the filter in this process
        os.environ["PYTHONWARNINGS"] = self._warnings  # Affects subprocesses

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = prepare_logger(value, class_name=self.__class__.__name__)

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, value):
        self._experiment = value
        if value:
            mlflow.set_experiment(value)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    @typechecked
    def random_state(self, value: Optional[int]):
        if value and value < 0:
            raise ValueError(
                "Invalid value for the random_state parameter. "
                f"Value should be >0, got {value}."
            )
        random.seed(value)  # Set random seed
        np.random.seed(value)
        self._random_state = value

    # Methods ====================================================== >>

    @staticmethod
    @typechecked
    def _prepare_input(X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Prepare the input data.

        Convert X and y to pandas (if not already) and perform standard
        compatibility checks (dimensions, length, indices, etc...).

        Parameters
        ----------
        X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Feature dataset.

        y: pd.Series
            Target column corresponding to X.

        """
        # If data has more than 2 dimensions and is not a text corpus,
        # create a dataframe with one multidimensional column
        array = np.array(X)
        if array.ndim > 2 and not isinstance(array[0, 0, 0], str):
            X = pd.DataFrame({"Multidimensional feature": [row for row in X]})
        else:
            X = to_df(deepcopy(X))  # Make copy to not overwrite mutable arguments

            # If text dataset, change the name of the column to Corpus
            if X.shape[1] == 1 and X[X.columns[0]].dtype == "object":
                X = X.rename(columns={X.columns[0]: "Corpus"})

        # Prepare target column
        if isinstance(y, (dict, *SEQUENCE)):
            if not isinstance(y, pd.Series):
                # Check that y is one-dimensional
                ndim = np.array(y).ndim
                if ndim != 1:
                    raise ValueError(f"y should be one-dimensional, got ndim={ndim}.")

                # Check X and y have the same number of rows
                if len(X) != len(y):
                    raise ValueError(
                        "X and y don't have the same number of rows,"
                        f" got len(X)={len(X)} and len(y)={len(y)}."
                    )

                y = to_series(y, index=X.index)

            elif not X.index.equals(y.index):  # Compare indices
                raise ValueError("X and y don't have the same indices!")

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError(f"Column {y} not found in X!")

            return X.drop(y, axis=1), X[y]

        elif isinstance(y, int):
            return X.drop(X.columns[y], axis=1), X[X.columns[y]]

        return X, y

    def _get_data_and_idx(self, arrays, y=-1, use_n_rows=True):
        """Get the dataset and indices from a sequence of indexables.

        Parameters
        ----------
        arrays: tuple of indexables
            Data set(s) provided. Should follow the API's input format.

        y: int, str or sequence, optional (default=-1)
            Target column corresponding to X.

        use_n_rows: bool, optional (default=True)
            Whether to use the `n_rows` parameter on the dataset.

        """

        def _no_train_test(data):
            """Path to follow when no train and test are provided."""
            if use_n_rows:
                if self.n_rows > len(data):
                    raise ValueError(
                        "Invalid value for the n_rows parameter. The provided"
                        " number is larger than the number of samples, got "
                        f"len(data)={len(data)} and n_rows={int(self.n_rows)}."
                    )

                # Select subset of the data
                n = self.n_rows if self.n_rows > 1 else len(data) * self.n_rows
                if self.shuffle:
                    data = data.sample(n=int(n), random_state=self.random_state)
                else:
                    data = data.iloc[:int(n), :]

            data = data.reset_index(drop=True)

            if len(data) < 2:
                raise ValueError(
                    "Invalid value for the n_rows parameter, got "
                    f"{self.n_rows}. Length of dataset can not be <2."
                )

            if self.test_size <= 0 or self.test_size >= len(data):
                raise ValueError(
                    "Invalid value for the test_size parameter. Value "
                    f"should lie between 0 and len(X), got {self.test_size}."
                )

            # Define train and test indices
            if self.test_size < 1:
                test_idx = int(self.test_size * len(data))
            else:
                test_idx = self.test_size
            idx = [len(data) - test_idx, test_idx]

            return data, idx

        def _has_train_test(train, test):
            """Path to follow when train and test are provided."""
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
                else:
                    raise ValueError(
                        "Invalid value for the n_rows parameter. Value has "
                        "to be <1 when a train and test set are provided."
                    )

            data = pd.concat([train, test]).reset_index(drop=True)
            idx = [len(train), len(test)]

            return data, idx

        # Process input arrays ===================================== >>

        if len(arrays) == 0:
            if self.branch.data is None:
                raise ValueError(
                    "The data arrays are empty! Provide the data to run the pipeline "
                    "successfully. See the documentation for the allowed formats."
                )
            else:
                return self.branch.data, self.branch.idx

        elif len(arrays) == 1:
            # arrays=(X,)
            data = merge(*self._prepare_input(arrays[0], y=y))
            data, idx = _no_train_test(data)

        elif len(arrays) == 2:
            if len(arrays[0]) == len(arrays[1]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test))
                train = merge(*self._prepare_input(arrays[0][0], arrays[0][1]))
                test = merge(*self._prepare_input(arrays[1][0], arrays[1][1]))
                data, idx = _has_train_test(train, test)
            elif isinstance(arrays[1], (int, str)) or np.array(arrays[1]).ndim == 1:
                # arrays=(X, y)
                data = merge(*self._prepare_input(arrays[0], arrays[1]))
                data, idx = _no_train_test(data)
            else:
                # arrays=(train, test)
                train = merge(*self._prepare_input(arrays[0], y=y))
                test = merge(*self._prepare_input(arrays[1], y=y))
                data, idx = _has_train_test(train, test)

        elif len(arrays) == 4:
            # arrays=(X_train, X_test, y_train, y_test)
            train = merge(*self._prepare_input(arrays[0], arrays[2]))
            test = merge(*self._prepare_input(arrays[1], arrays[3]))
            data, idx = _has_train_test(train, test)

        else:
            raise ValueError(
                "Invalid data arrays. See the documentation for the allowed formats."
            )

        return data, idx

    @composed(crash, typechecked)
    def log(self, msg: Union[int, float, str], level: int = 0):
        """Print and save output to log file.

        Parameters
        ----------
        msg: int, float or str
            Message to save to the logger and print to stdout.

        level: int, optional (default=0)
            Minimum verbosity level to print the message.
            If 42, don't save to log.

        """
        if self.verbose >= level:
            print(msg)

        if self.logger is not None and level != 42:
            for text in str(msg).split("\n"):
                self.logger.info(str(text))

    @composed(crash, method_to_log, typechecked)
    def save(self, filename: str = "auto", save_data: bool = True):
        """Save the instance to a pickle file.

        Parameters
        ----------
        filename: str, optional (default="auto")
            Name of the file. Use "auto" for automatic naming.

        save_data: bool, optional (default=True)
            Whether to save the dataset with the instance. This
            parameter is ignored if the method is not called from
            a trainer.

        """
        if not save_data and hasattr(self, "dataset"):
            data = {}  # Store the data to reattach later
            for key, value in self._branches.items():
                data[key] = deepcopy(value.data)
                value.data = None

        if filename.endswith("auto"):
            filename = filename.replace("auto", self.__class__.__name__)

        with open(filename, "wb") as f:
            dill.dump(self, f)  # Dill replaces pickle to dump lambdas

        # Restore the data to the attributes
        if not save_data and hasattr(self, "dataset"):
            for key, value in self._branches.items():
                value.data = data[key]

        self.log(self.__class__.__name__ + " saved successfully!", 1)
