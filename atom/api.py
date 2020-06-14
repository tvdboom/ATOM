# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the API classes

"""

# << ============ Import Packages ============ >>

# Standard packages
from typeguard import typechecked
from typing import Optional, Union

# Own modules
from .atom import ATOM
from .utils import X_TYPES, Y_TYPES, prepare_logger


# << ================= Classes ================= >>

class ATOMClassifier(ATOM):
    """ATOM class for classification tasks."""

    @typechecked
    def __init__(self,
                 X: X_TYPES,
                 y: Y_TYPES = -1,
                 n_rows: Union[int, float] = 1,
                 test_size: float = 0.3,
                 logger: Optional[Union[str, callable]] = None,
                 n_jobs: int = 1,
                 warnings: Union[bool, str] = True,
                 verbose: int = 0,
                 random_state: Optional[int] = None):
        """Assign the algorithm's goal and prepare the log file.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Dataset containing the features, with shape=(n_samples, n_features)

        y: int, str, sequence, np.array or pd.Series, optional (default=-1)
            - If int: index of the column of X which is selected as target
            - If string: value_name of the target column in X
            - Else: data target column with shape=(n_samples,)

        n_rows: int or float, optional (default=1)
            if <=1: fraction of the data to use.
            if >1: number of rows to use.

        test_size: float, optional (default=0.3)
            Split fraction of the train and test set.

        n_jobs: int, optional (default=1)
            Number of cores to use for parallel processing.
                - If -1, use all available cores
                - If <-1, use available_cores - 1 + value

            Beware that using multiple processes on the same machine may
            cause memory issues for large datasets.

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything.
                - 1 to print basic information.
                - 2 to print extended information.

        warnings: bool or str, optional (default=True)
            - If boolean: True for showing all warnings (equal to 'default'
                          when string) and False for suppressing them (equal
                          to 'ignore' when string).
            - If string: one of python's actions in the warnings environment.

            Note that this changing this parameter will affect the
            `PYTHONWARNINGS` environment.

        logger: str, callable or None, optional (default=None)
            - If string: name of the logging file. 'auto' for default name
                         with timestamp. None to not save any log.
            - If callable: python Logger object.

        random_state: int or None, optional (default=None)
            Seed used by the random number generator. If None, the random
            number generator is the RandomState instance used by `np.random`.

        """
        self.goal = 'classification'
        self.logger = prepare_logger(logger, self.__class__.__name__)
        super().__init__(X, y,
                         n_rows=n_rows,
                         test_size=test_size,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         warnings=warnings,
                         random_state=random_state)


class ATOMRegressor(ATOM):
    """ATOM class for regression tasks."""

    @typechecked
    def __init__(self,
                 X: X_TYPES,
                 y: Y_TYPES = -1,
                 n_rows: Union[int, float] = 1,
                 test_size: float = 0.3,
                 n_jobs: int = 1,
                 warnings: Union[bool, str] = True,
                 verbose: int = 0,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        """Assign the algorithm's goal and prepare the log file.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Dataset containing the features, with shape=(n_samples, n_features)

        y: int, str, sequence, np.array or pd.Series, optional (default=-1)
            - If int: index of the column of X which is selected as target
            - If string: value_name of the target column in X
            - Else: data target column with shape=(n_samples,)

        n_rows: int or float, optional (default=1)
            if <=1: fraction of the data to use.
            if >1: number of rows to use.

        test_size: float, optional (default=0.3)
            Split fraction of the train and test set.

        n_jobs: int, optional (default=1)
            Number of cores to use for parallel processing.
                - If -1, use all available cores
                - If <-1, use available_cores - 1 + value

            Beware that using multiple processes on the same machine may
            cause memory issues for large datasets.

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything.
                - 1 to print basic information.
                - 2 to print extended information.

        warnings: bool or str, optional (default=True)
            - If boolean: True for showing all warnings (equal to 'default'
                          when string) and False for suppressing them (equal
                          to 'ignore' when string).
            - If string: one of python's actions in the warnings environment.

            Note that this changing this parameter will affect the
            `PYTHONWARNINGS` environment.

        logger: str, callable or None, optional (default=None)
            - If string: name of the logging file. 'auto' for default name
                         with timestamp. None to not save any log.
            - If callable: python Logger object.

        random_state: int or None, optional (default=None)
            Seed used by the random number generator. If None, the random
            number generator is the RandomState instance used by `np.random`.

        """
        self.goal = 'regression'
        self.logger = prepare_logger(logger, self.__class__.__name__)
        super().__init__(X, y,
                         n_rows=n_rows,
                         test_size=test_size,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         warnings=warnings,
                         random_state=random_state)
