# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the API classes.

"""

# Standard packages
import pickle
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union, Tuple

# Own modules
from .atom import ATOM
from .basetransformer import BaseTransformer
from .utils import X_TYPES, Y_TYPES, merge


# Functions ================================================================= >>

@typechecked
def ATOMLoader(filename: str,
               data: Optional[Union[X_TYPES, Tuple]] = None,
               transform_data: bool = True,
               verbose: int = 0):
    """Load a class from a pickle file.

    If its a training instance, you can load new data.
    If its an ATOM instance, you can load new data and apply all data
    transformations in the pipeline.

    Parameters
    ----------
    filename: str
        Name of the pickle file to load.

    data: dict, sequence, np.array, pd.DataFrame or tuple, optional (default=None)
        Dataset containing the features, with shape=(n_samples, n_features) or
        tuple of the form (X, y).

    transform_data: bool, optional (default=True)
        Only if the loaded file is an ATOM instance.

    verbose: int, optional (default=0)
        Verbosity level of the transformations applied on the new data.

    """
    # Check verbose parameter
    if verbose < 0 or verbose > 2:
        raise ValueError("Invalid value for the verbose parameter." +
                         f"Value should be between 0 and 2, got {verbose}.")

    with open(filename, 'rb') as f:
        cls_ = pickle.load(f)

    if data is not None:
        if not hasattr(cls_, '_data'):
            raise TypeError("Data is provided but the class is not an ATOM nor " +
                            f"training instance, got {cls_.__class__.__name__}.")

        elif cls_._data is not None:
            raise ValueError("The loaded {} instance already contains data!"
                             .format(cls_.__class__.__name__))

        # Get X and y from data
        if isinstance(data, (list, tuple)):
            X, y = data[0], data[1]
        else:
            X, y = data, None

        X, y = BaseTransformer._prepare_input(X, y)
        cls_._data = X if y is None else merge(X, y)

        if hasattr(cls_, 'pipeline') and transform_data:
            # Transform the data through all steps in the pipeline
            for estimator in cls_.pipeline:
                estimator.set_params(verbose=verbose)

                # Some transformations are only applied on the training set
                if estimator.__class__.__name__ in ['Outliers', 'Balancer']:
                    X_train, y_train = estimator.transform(cls_.X_train, cls_.y_train)
                    cls_._data = pd.concat([merge(X_train, y_train), cls_.test])
                    cls_._data.reset_index(drop=True, inplace=True)
                else:
                    X = estimator.transform(cls_.X, cls_.y)
                    if isinstance(X, tuple):  # Estimator returned X, y
                        X = merge(*X)
                    cls_._data = X.reset_index(drop=True)

        if getattr(cls_, 'trainer', None):
            cls_.trainer._data = cls_._data

    return cls_


# Classes =================================================================== >>

class ATOMClassifier(BaseTransformer, ATOM):
    """ATOM class for classification tasks.

    Parameters
    ----------
    X: dict, sequence, np.array or pd.DataFrame
        Dataset containing the features, with shape=(n_samples, n_features)

    y: int, str, sequence, np.array or pd.Series, optional (default=-1)
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Data target column with shape=(n_samples,).

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the data to use.
        - If >1: Number of rows to use.

    test_size: float, optional (default=0.2)
        Split fraction for the training and test set.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - n_jobs.

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

        Note that ATOM can't manage warnings that go directly from C++ code
        to the stdout/stderr.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name, False for no logger.
        - If string: name of the logging file. 'auto' for default name.
        - If class: python Logger object.

        Note that warnings will not be saved to the logger in any case.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    @typechecked
    def __init__(self,
                 X: X_TYPES,
                 y: Y_TYPES = -1,
                 n_rows: Union[int, float] = 1,
                 test_size: float = 0.2,
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
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Data target column with shape=(n_samples,).

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the data to use.
        - If >1: Number of rows to use.

    test_size: float, optional (default=0.2)
        Split fraction for the training and test set.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - n_jobs.

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

        Note that ATOM can't manage warnings that go directly from C++ code
        to the stdout/stderr.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name, False for no logger.
        - If string: name of the logging file. 'auto' for default name.
        - If class: python Logger object.

        Note that warnings will not be saved to the logger in any case.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    @typechecked
    def __init__(self,
                 X: X_TYPES,
                 y: Y_TYPES = -1,
                 n_rows: Union[int, float] = 1,
                 test_size: float = 0.2,
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
