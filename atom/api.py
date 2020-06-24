# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the API classes.

"""

# Standard packages
from typeguard import typechecked
from typing import Optional, Union

# Own modules
from .atom import ATOM
from .basetransformer import BaseTransformer
from .utils import X_TYPES, Y_TYPES, infer_task


# Classes =================================================================== >>

class ATOMClassifier(BaseTransformer, ATOM):
    """ATOM class for classification tasks.

    Parameters
    ----------
    X: dict, sequence, np.array or pd.DataFrame
        Dataset containing the features, with shape=(n_samples, n_features)

    y: int, str, sequence, np.array or pd.Series, optional (default=-1)
        - If int: Index of the target column in X.
        - If str: Name of the target column in X.
        - Else: Data target column with shape=(n_samples,).

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the data to use.
        - If >1: Number of rows to use.

    test_size: float, optional (default=0.3)
        Split fraction for the training and test set.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - value.

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    warnings: bool or str, optional (default=True)
        - If True: Default warning action (equal to 'default' when string).
        - If False: Suppress all warnings (equal to 'ignore' when string).
        - If str: One of the possible actions in python's warnings environment.

        Note that changing this parameter will affect the `PYTHONWARNINGS`
        environment.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. 'auto' to create an automatic name.
        - If class: python Logger object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

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
        """Assign the algorithm's goal and prepare the standard properties."""
        self.goal = 'classification'
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         warnings=warnings,
                         logger=logger,
                         random_state=random_state)

        ATOM.__init__(self, X, y, n_rows, test_size)


class ATOMRegressor(BaseTransformer, ATOM):
    """ATOM class for regression tasks.

    Parameters
    ----------
    X: dict, sequence, np.array or pd.DataFrame
        Dataset containing the features, with shape=(n_samples, n_features)

    y: int, str, sequence, np.array or pd.Series, optional (default=-1)
        - If int: Index of the target column in X.
        - If str: Name of the target column in X.
        - Else: Data target column with shape=(n_samples,).

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the data to use.
        - If >1: Number of rows to use.

    test_size: float, optional (default=0.3)
        Split fraction for the training and test set.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - value.

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    warnings: bool or str, optional (default=True)
        - If True: Default warning action (equal to 'default' when string).
        - If False: Suppress all warnings (equal to 'ignore' when string).
        - If str: One of the possible actions in python's warnings environment.

        Note that changing this parameter will affect the `PYTHONWARNINGS`
        environment.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. 'auto' to create an automatic name.
        - If class: python Logger object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

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
        """Assign the algorithm's goal and prepare the standard properties."""
        self.goal = 'regression'
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         warnings=warnings,
                         logger=logger,
                         random_state=random_state)

        ATOM.__init__(self, X, y, n_rows, test_size)
