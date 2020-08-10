# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the BaseTransformer class.

"""

# Standard packages
import os
import pickle
import numpy as np
import pandas as pd
import multiprocessing
import warnings as warn
from copy import deepcopy
from typeguard import typechecked
from typing import Union, Optional

# Own modules
from .utils import prepare_logger, to_df, to_series, composed, method_to_log, crash


class BaseTransformer(object):
    """Base estimator for classes in the package.

    Contains shared properties (n_jobs, verbose, warnings, logger, random_state)
    and standard methods across multiple classes.

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

    def __init__(self, **kwargs):
        """Update the properties with the provided kwargs."""
        for attr in ['n_jobs', 'verbose', 'warnings', 'logger', 'random_state']:
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
                    f"Invalid value for the n_jobs parameter, got {n_jobs}.", 1)

        self._n_jobs = n_jobs

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    @typechecked
    def verbose(self, verbose: int):
        if verbose < 0 or verbose > 2:
            raise ValueError("Invalid value for the verbose parameter. Value" +
                             f" should be between 0 and 2, got {verbose}.")
        self._verbose = verbose

    @property
    def warnings(self):
        return self._warnings

    @warnings.setter
    @typechecked
    def warnings(self, warnings: Union[bool, str]):
        if isinstance(warnings, bool):
            self._warnings = 'default' if warnings else 'ignore'
        else:
            opts = ['error', 'ignore', 'always', 'default', 'module', 'once']
            if warnings not in opts:
                raise ValueError(
                    "Invalid value for the warnings parameter, got "
                    f"{warnings}. Choose from: {', '.join(opts)}.")
            self._warnings = warnings

        warn.simplefilter(self._warnings)  # Change the filter in this process
        os.environ['PYTHONWARNINGS'] = self._warnings  # Affects subprocesses

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
            raise ValueError("Invalid value for the random_state parameter. " +
                             f"Value should be >0, got {random_state}.")
        np.random.seed(random_state)  # Set random seed
        self._random_state = random_state

    # Methods =============================================================== >>

    @staticmethod
    def _prepare_input(X, y):
        """Prepare the input data.

        Copy X and y and convert to pandas. If already in pandas frame, reset
        all indices for them to be able to merge.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None, y is not used in the estimator
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Copy of the feature dataset.

        y: pd.Series
            Copy of the target column corresponding to X.

        """
        X = to_df(deepcopy(X))  # Copy to not overwrite mutable variables

        # Convert array to dataframe and target column to pandas series
        if isinstance(y, (list, tuple, dict, np.ndarray, pd.Series)):
            y = deepcopy(y)
            if len(X) != len(y):
                raise ValueError("X and y don't have the same number of " +
                                 f"rows: {len(X)}, {len(y)}.")

            # Convert y to pd.Series
            if not isinstance(y, pd.Series):
                if not isinstance(y, np.ndarray):
                    y = np.array(y)

                # Check that y is one-dimensional
                if y.ndim != 1:
                    raise ValueError(
                        f"y should be one-dimensional, got y.ndim={y.ndim}.")
                y = to_series(y, index=X.index)

            elif not X.index.equals(y.index):  # Compare indices
                raise ValueError("X and y don't have the same indices!")

            return X, y

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError("Target column not found in X!")

            return X.drop(y, axis=1), X[y]

        elif isinstance(y, int):
            return X.drop(X.columns[y], axis=1), X[X.columns[y]]

        elif y is None:
            return X, y

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
                while msg.startswith('\n'):  # Insert empty lines
                    self.logger.info('')
                    msg = msg[1:]
            self.logger.info(str(msg))

    @composed(crash, method_to_log, typechecked)
    def save(self, filename: Optional[str] = None, **kwargs):
        """Save the class to a pickle file.

        Parameters
        ----------
        filename: str or None, optional (default=None)
            Name to save the file with. None or 'auto' to save with default name.

        **kwargs
            Additional keyword arguments. Can contain:
                - 'save_data': Whether to save the dataset with the instance.
                               Only for ATOM or a training class.

        """
        if kwargs.get('save_data') is False and hasattr(self, '_data'):
            data = self._data.copy()  # Store the data to reattach later
            self._data = None
            if getattr(self, 'trainer', None):
                self.trainer._data = None

        if not filename:
            filename = self.__class__.__name__
        elif filename == 'auto' or filename.endswith('/auto'):
            filename = filename.replace('auto', self.__class__.__name__)

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

        if kwargs.get('save_data') is False and hasattr(self, '_data'):
            self._data = data

        self.log(self.__class__.__name__ + " saved successfully!", 1)
