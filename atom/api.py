# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the API classes

"""

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union, Sequence

# Own modules
from .atom import ATOM
from .utils import X_types, y_types, prepare_logger


# << ================= Classes ================= >>

class ATOMClassifier(ATOM):
    """ATOM class for classification tasks."""

    @typechecked
    def __init__(self,
                 X: X_types,
                 y: y_types = -1,
                 percentage: Union[int, float] = 100,
                 test_size: float = 0.3,
                 log: Optional[str] = None,
                 n_jobs: int = 1,
                 warnings: bool = False,
                 verbose: int = 0,
                 random_state: Optional[int] = None):
        """Class initializer.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Dataset containing the features, with shape=(n_samples, n_features)

        y: int, str, sequence, np.array or pd.Series, optional (default=-1)
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        percentage: int or float, optional (default=100)
            Percentage of the data to use in the pipeline.

        test_size: float, optional (default=0.3)
            Split fraction of the train and test set.

        log: string or None, optional (default=None)
            Name of the log file. 'auto' for default name with date and time.
            None to not save any log.

        n_jobs: int, optional (default=1)
            Number of cores to use for parallel processing.
                - If -1, use all available cores
                - If <-1, use available_cores - 1 + n_jobs

            Beware that using multiple processes on the same machine may cause
            memory issues for large datasets.

        warnings: bool, optional (default=False)
            whether to show warnings when fitting the models.

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything
                - 1 to print minimum information
                - 2 to print average information
                - 3 to print maximum information

        random_state: int or None, optional (default=None)
            Seed used by the random number generator. If None, the random
            number generator is the RandomState instance used by `np.random`.

        """
        self.log = prepare_logger(log)
        self.goal = 'classification'
        super().__init__(X, y,
                         percentage=percentage,
                         test_size=test_size,
                         n_jobs=n_jobs,
                         warnings=warnings,
                         verbose=verbose,
                         random_state=random_state)


class ATOMRegressor(ATOM):
    """ATOM class for regression tasks."""

    @typechecked
    def __init__(self,
                 X: X_types,
                 y: y_types = -1,
                 percentage: Union[int, float] = 100,
                 test_size: float = 0.3,
                 n_jobs: int = 1,
                 warnings: bool = False,
                 verbose: int = 0,
                 log: Optional[str] = None,
                 random_state: Optional[int] = None):
        """Class initializer.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Dataset containing the features, with shape=(n_samples, n_features)

        y: int, str, sequence, np.array or pd.Series, optional (default=-1)
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        percentage: int or float, optional (default=100)
            Percentage of the data to use in the pipeline.

        test_size: float, optional (default=0.3)
            Split fraction of the train and test set.

        n_jobs: int, optional (default=1)
            Number of cores to use for parallel processing.
                - If -1, use all available cores
                - If <-1, use available_cores - 1 + n_jobs

            Beware that using multiple processes on the same machine may cause
            memory issues for large datasets.

        warnings: bool, optional (default=False)
            whether to show warnings when fitting the models.

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything
                - 1 to print minimum information
                - 2 to print average information
                - 3 to print maximum information

        log: string or None, optional (default=None)
            Name of the log file. 'auto' for default name with date and time.
            None to not save any log.

        random_state: int or None, optional (default=None)
            Seed used by the random number generator. If None, the random
            number generator is the RandomState instance used by `np.random`.

        """
        self.log = prepare_logger(log)
        self.goal = 'regression'
        super().__init__(X, y,
                         percentage=percentage,
                         test_size=test_size,
                         n_jobs=n_jobs,
                         warnings=warnings,
                         verbose=verbose,
                         random_state=random_state)
