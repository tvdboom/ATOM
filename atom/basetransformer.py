# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the BaseTransformer class.

"""

# Standard packages
import os
import pickle
import random
import numpy as np
import pandas as pd
import multiprocessing
import warnings as warn
from copy import deepcopy
from typeguard import typechecked
from typing import Union, Optional

# Own modules
from .utils import (
    ARRAY_TYPES, X_TYPES, Y_TYPES, to_df, to_series, merge,
    prepare_logger, composed, method_to_log, crash
)


class BaseTransformer(object):
    """Base estimator for classes in the package.

    Contains shared properties (n_jobs, verbose, warnings, logger, random_state)
    and standard methods across transformers.

    Parameters
    ----------
    **kwargs
        Standard keyword arguments for the classes. Can include:
            - n_jobs: Number of cores to use for parallel processing.
            - verbose: Verbosity level of the output.
            - warnings: Whether to show or suppress encountered warnings.
            - logger: Name of the logging file or Logger object.
            - random_state: Seed used by the random number generator.

    """

    attrs = ["n_jobs", "verbose", "warnings", "logger", "random_state"]

    def __init__(self, **kwargs):
        """Update the properties with the provided kwargs."""
        for attr in self.attrs:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])

    # Properties ============================================================ >>

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    @typechecked
    def n_jobs(self, n_jobs: int):
        # Check number of cores for multiprocessing
        n_cores = multiprocessing.cpu_count()
        if n_jobs > n_cores:
            n_jobs = n_cores
        else:
            n_jobs = n_cores + 1 + n_jobs if n_jobs < 0 else n_jobs

            # Final check for negative input
            if n_jobs < 1:
                raise ValueError(
                    f"Invalid value for the n_jobs parameter, got {n_jobs}.", 1
                )

        self._n_jobs = n_jobs

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    @typechecked
    def verbose(self, verbose: int):
        if verbose < 0 or verbose > 2:
            raise ValueError(
                "Invalid value for the verbose parameter. Value"
                f" should be between 0 and 2, got {verbose}."
            )
        self._verbose = verbose

    @property
    def warnings(self):
        return self._warnings

    @warnings.setter
    @typechecked
    def warnings(self, warnings: Union[bool, str]):
        if isinstance(warnings, bool):
            self._warnings = "default" if warnings else "ignore"
        else:
            opts = ["error", "ignore", "always", "default", "module", "once"]
            if warnings not in opts:
                raise ValueError(
                    "Invalid value for the warnings parameter, got "
                    f"{warnings}. Choose from: {', '.join(opts)}."
                )
            self._warnings = warnings

        warn.simplefilter(self._warnings)  # Change the filter in this process
        os.environ["PYTHONWARNINGS"] = self._warnings  # Affects subprocesses

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = prepare_logger(logger, class_name=self.__class__.__name__)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    @typechecked
    def random_state(self, random_state: Optional[int]):
        if random_state and random_state < 0:
            raise ValueError(
                "Invalid value for the random_state parameter. "
                f"Value should be >0, got {random_state}."
            )
        random.seed(random_state)  # Set random seed
        np.random.seed(random_state)
        self._random_state = random_state

    # Methods =============================================================== >>

    @staticmethod
    @typechecked
    def _prepare_input(X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Prepare the input data.

        Copy X and y, convert to pandas (if not already) and perform standard
        compatibility checks (dimensions, length, indices, etc...).

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Dataset containing the features, with shape=(n_samples, n_features).

        y: int, str, array-like or None, optional (default=None)
            - If None, y is not used in the estimator.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Copy of the feature dataset.

        y: pd.Series
            Copy of the target column corresponding to X.

        """
        X = to_df(deepcopy(X))  # Copy to not overwrite mutable variables

        # Convert target column to series
        if isinstance(y, (list, tuple, dict, np.ndarray, pd.Series)):
            y = deepcopy(y)

            # Convert y to pd.Series
            if not isinstance(y, pd.Series):
                if not isinstance(y, np.ndarray):
                    y = np.array(y)

                # Check that y is one-dimensional
                if y.ndim != 1:
                    raise ValueError(
                        f"y should be one-dimensional, got y.ndim={y.ndim}."
                    )

                # Check X and y have the same number of rows
                if len(X) != len(y):
                    raise ValueError(
                        "X and y don't have the same number of rows,"
                        f" got len(X)={len(X)} and len(y)={len(y)}."
                    )

                y = to_series(y, index=X.index)

            elif not X.index.equals(y.index):  # Compare indices
                raise ValueError("X and y don't have the same indices!")

            return X, y

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError(f"Column {y} not found in X!")

            return X.drop(y, axis=1), X[y]

        elif isinstance(y, int):
            return X.drop(X.columns[y], axis=1), X[X.columns[y]]

        else:  # y is None
            return X, y

    def _get_data_and_idx(self, arrays, use_n_rows=True):
        """Get the dataset and indices from a sequence of indexables.

        Parameters
        ----------
        arrays: tuple of indexables
            Dataset(s) provided. Formats according to the API's input format.

        use_n_rows: bool, optional (default=True)
            Whether to use the n_rows parameter on the dataset.

        """
        try:
            if len(arrays) == 1:
                # arrays=(X,)
                data, _ = self._prepare_input(arrays[0])
            elif len(arrays) == 2:
                # arrays=(X, y)
                data = merge(*self._prepare_input(arrays[0], arrays[1]))

            # Select subsample while shuffling the dataset
            if use_n_rows:
                if self.n_rows <= 1:
                    kwargs = dict(frac=self.n_rows, random_state=self.random_state)
                else:
                    kwargs = dict(n=int(self.n_rows), random_state=self.random_state)
                data = data.sample(**kwargs)

            # Reset all indices
            data.reset_index(drop=True, inplace=True)

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

        except (UnboundLocalError, TypeError, ValueError, AttributeError):
            if len(arrays) == 0:
                raise ValueError(
                    "The data arrays are empty! Provide the data to run the pipeline "
                    "successfully. See the documentation for the allowed formats."
                )
            elif isinstance(arrays[0], ARRAY_TYPES) and len(arrays[0]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test))
                train = merge(*self._prepare_input(arrays[0][0], arrays[0][1]))
                test = merge(*self._prepare_input(arrays[1][0], arrays[1][1]))
            elif len(arrays) == 2:
                # arrays=(train, test)
                train, _ = self._prepare_input(arrays[0])
                test, _ = self._prepare_input(arrays[1])
            elif len(arrays) == 4:
                # arrays=(X_train, X_test, y_train, y_test)
                train = merge(*self._prepare_input(arrays[0], arrays[2]))
                test = merge(*self._prepare_input(arrays[1], arrays[3]))
            else:
                raise ValueError(
                    "Invalid data arrays. See the documentation "
                    "for the allowed formats."
                )

            # Skip this if called from training instance
            if hasattr(self, "n_rows") and use_n_rows:
                # Select same subsample of train and test set
                if self.n_rows <= 1:
                    kwargs = dict(frac=self.n_rows, random_state=self.random_state)
                    train = train.sample(**kwargs)
                    test = test.sample(**kwargs)
                else:
                    raise ValueError(
                        "Invalid value for the n_rows parameter. value "
                        "should be <=1 when train and test are provided."
                    )

            # Update the data and reset the indices
            data = pd.concat([train, test]).reset_index(drop=True)
            idx = [len(train), len(test)]

        return data, idx

    @composed(crash, typechecked)
    def log(self, msg: Union[int, float, str], level: int = 0):
        """Print and save output to log file.

        Parameters
        ----------
        msg: int, float or str
            Message to save to the logger and print to stdout.

        level: int
            Minimum verbosity level in order to print the message.
            If 42, don't save to log.

        """
        if self.verbose >= level:
            print(msg)

        if self.logger is not None and level != 42:
            if isinstance(msg, str):
                while msg.startswith("\n"):  # Insert empty lines
                    self.logger.info("")
                    msg = msg[1:]
            self.logger.info(str(msg))

    @composed(crash, method_to_log, typechecked)
    def save(self, filename: Optional[str] = None, **kwargs):
        """Save the class to a pickle file.

        Parameters
        ----------
        filename: str or None, optional (default=None)
            Name to save the file with. None or "auto" to save with default name.

        **kwargs
            Additional keyword arguments. Can contain:
                - "save_data": Whether to save the dataset with the instance.
                               Only for `training` instances.

        """
        if kwargs.get("save_data") is False and hasattr(self, "_data"):
            data = self._data.copy()  # Store the data to reattach later
            self._data = None
            if getattr(self, "trainer", None):
                self.trainer._data = None

        if not filename:
            filename = self.__class__.__name__
        elif filename == "auto" or filename.endswith("/auto"):
            filename = filename.replace("auto", self.__class__.__name__)

        with open(filename, "wb") as file:
            pickle.dump(self, file)

        if kwargs.get("save_data") is False and hasattr(self, "_data"):
            self._data = data

        self.log(self.__class__.__name__ + " saved successfully!", 1)
