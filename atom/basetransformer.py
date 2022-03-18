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
from typing import Optional, Union

import dill
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typeguard import typechecked

from .utils import (
    SEQUENCE, X_TYPES, Y_TYPES, composed, crash, lst, merge, method_to_log,
    prepare_logger, to_df, to_series,
)


class BaseTransformer:
    """Base class for estimators in the package.

    Contains shared properties and standard methods across transformers.

    Parameters
    ----------
    **kwargs
        Standard keyword arguments for the classes. Can include:
            - n_jobs: Number of cores to use for parallel processing.
            - verbose: Verbosity level of the output.
            - warnings: Whether to show or suppress encountered warnings.
            - logger: Name of the log file or Logger object.
            - experiment: Name of the mlflow experiment used for tracking.
            - gpu: Whether to train the pipeline on GPU.
            - random_state: Seed used by the random number generator.

    """

    attrs = [
        "n_jobs",
        "verbose",
        "warnings",
        "logger",
        "experiment",
        "gpu",
        "random_state",
    ]

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
    def gpu(self):
        return self._gpu

    @gpu.setter
    @typechecked
    def gpu(self, value: Union[bool, str]):
        if isinstance(value, str) and value.lower() != "force":
            raise ValueError(
                "Invalid value for the gpu parameter. Only True, "
                f"False and 'force' are valid values, got {value}."
            )
        self._gpu = value

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

    def _get_gpu(self, est):
        """Get a GPU or CPU estimator depending on availability.

        Parameters
        ----------
        est: estimator
            Class to get GPU implementation from.

        Returns
        -------
        estimator
            Provided estimator or cuml implementation.

        """
        if self.gpu:
            try:
                import cuml
                return getattr(cuml, est.__name__)
            except ModuleNotFoundError:
                if self.gpu == "force":
                    raise ModuleNotFoundError(
                        "It looks like cuml is not installed. Refer to the package's "
                        "documentation to learn how to utilize the machine's GPU."
                    )
                else:
                    self.log(
                        f"Unable to import cuml.{est.__name__}. "
                        "Using CPU implementation instead.", 1
                    )

        return est

    @staticmethod
    @typechecked
    def _prepare_input(X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Prepare the input data.

        Convert X and y to pandas (if not already) and perform standard
        compatibility checks (dimensions, length, indices, etc...).

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        pd.DataFrame
            Feature dataset.

        pd.Series
            Target column corresponding to X.

        """
        # If data has more than 2 dimensions and is not a text corpus,
        # create a dataframe with one multidimensional column
        array = np.array(X)
        if array.ndim > 2 and not isinstance(array[0, 0, 0], str):
            X = pd.DataFrame({"multidim feature": [row for row in X]})
        else:
            X = to_df(deepcopy(X))  # Make copy to not overwrite mutable arguments

            # If text dataset, change the name of the column to corpus
            if list(X.columns) == ["feature_1"] and X[X.columns[0]].dtype == "object":
                X = X.rename(columns={X.columns[0]: "corpus"})

            # Convert all column names to str
            X.columns = [str(col) for col in X.columns]

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

    def _set_index(self, df):
        """Assign an index to the dataframe."""
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

    def _get_stratify_columns(self, df):
        """Get columns to stratify by."""
        # Stratification is not possible when the data can not change order
        if self.shuffle is False or self.stratify is False:
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

    def _get_data(self, arrays, y=-1, use_n_rows=True):
        """Get the data sets and indices from a sequence of indexables.

        Parameters
        ----------
        arrays: tuple of indexables
            Data set(s) provided. Should follow the API's input format.

        y: int, str or sequence, optional (default=-1)
            Target column corresponding to X.

        use_n_rows: bool, optional (default=True)
            Whether to use the `n_rows` parameter on the dataset.

        Returns
        -------
        pd.DataFrame
            Dataset containing the train and test sets.

        list
            Sizes of the train and test sets.

        pd.DataFrame or None
            Holdout data set. Returns None if not specified.

        """

        def _no_data_sets(data):
            """Path to follow when no data sets are provided."""
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

        def _has_data_sets(train, test, holdout=None):
            """Path to follow when data sets are provided."""
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
            data, idx, holdout = _no_data_sets(data)

        elif len(arrays) == 2:
            if isinstance(arrays[0], tuple) and len(arrays[0]) == len(arrays[1]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test))
                train = merge(*self._prepare_input(arrays[0][0], arrays[0][1]))
                test = merge(*self._prepare_input(arrays[1][0], arrays[1][1]))
                data, idx, holdout = _has_data_sets(train, test)
            elif isinstance(arrays[1], (int, str)) or np.array(arrays[1]).ndim == 1:
                # arrays=(X, y)
                data = merge(*self._prepare_input(arrays[0], arrays[1]))
                data, idx, holdout = _no_data_sets(data)
            else:
                # arrays=(train, test)
                train = merge(*self._prepare_input(arrays[0], y=y))
                test = merge(*self._prepare_input(arrays[1], y=y))
                data, idx, holdout = _has_data_sets(train, test)

        elif len(arrays) == 3:
            if len(arrays[0]) == len(arrays[1]) == len(arrays[2]) == 2:
                # arrays=((X_train, y_train), (X_test, y_test), (X_holdout, y_holdout))
                train = merge(*self._prepare_input(arrays[0][0], arrays[0][1]))
                test = merge(*self._prepare_input(arrays[1][0], arrays[1][1]))
                holdout = merge(*self._prepare_input(arrays[2][0], arrays[2][1]))
                data, idx, holdout = _has_data_sets(train, test, holdout)
            else:
                # arrays=(train, test, holdout)
                train = merge(*self._prepare_input(arrays[0], y=y))
                test = merge(*self._prepare_input(arrays[1], y=y))
                holdout = merge(*self._prepare_input(arrays[2], y=y))
                data, idx, holdout = _has_data_sets(train, test, holdout)

        elif len(arrays) == 4:
            # arrays=(X_train, X_test, y_train, y_test)
            train = merge(*self._prepare_input(arrays[0], arrays[2]))
            test = merge(*self._prepare_input(arrays[1], arrays[3]))
            data, idx, holdout = _has_data_sets(train, test)

        elif len(arrays) == 6:
            # arrays=(X_train, X_test, X_holdout, y_train, y_test, y_holdout)
            train = merge(*self._prepare_input(arrays[0], arrays[3]))
            test = merge(*self._prepare_input(arrays[1], arrays[4]))
            holdout = merge(*self._prepare_input(arrays[2], arrays[5]))
            data, idx, holdout = _has_data_sets(train, test, holdout)

        else:
            raise ValueError(
                "Invalid data arrays. See the documentation for the allowed formats."
            )

        return data, idx, holdout

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
            data = {"holdout": deepcopy(self.holdout)}  # Store data to reattach later
            self.holdout = None
            for key, value in self._branches.items():
                data[key] = deepcopy(value._data)
                value._data = None

        if filename.endswith("auto"):
            filename = filename.replace("auto", self.__class__.__name__)

        with open(filename, "wb") as f:
            dill.dump(self, f)  # Dill replaces pickle to dump lambdas

        # Restore the data to the attributes
        if not save_data and hasattr(self, "dataset"):
            self.holdout = data["holdout"]
            for key, value in self._branches.items():
                value._data = data[key]

        self.log(f"{self.__class__.__name__} successfully saved.", 1)
