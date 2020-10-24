# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the API classes.

"""

# Standard packages
import pickle
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union

# Own modules
from .atom import ATOM
from .basetransformer import BaseTransformer
from .utils import X_TYPES, Y_TYPES, merge


# Functions ================================================================= >>

@typechecked
def ATOMModel(estimator,
              name: str = None,
              longname: str = None,
              needs_scaling: bool = True,
              type: str = 'kernel'):
    """Convert an estimator to a model that can be ingested by ATOM's pipeline.

    Add the relevant attributes to the estimator so that they can be used
    when initializing the CustomModel class.

    Parameters
    ----------
    estimator: class
        Model's estimator. Can be a class or an instance.

    name: str, optional (default=None)
        Model's acronym. Used to call the `model` from the training instance.
        If None, the estimator's name will be used (not recommended).

    longname: str, optional (default=None)
        Full model's name. If None, the estimator's name will be used.

    needs_scaling: bool, optional (default=True)
        Whether the model needs scaled features.

    type: str, optional (default='kernel')
        Model's type. Choose from:
            - 'linear' for linear models.
            - 'tree' for tree-based models.
            - 'kernel' for the remaining models.

    """
    if name:
        estimator.name = name
    if longname:
        estimator.longname = longname
    estimator.needs_scaling = needs_scaling
    if type not in ('linear', 'tree', 'kernel'):
        raise ValueError("Invalid value for the type parameter. "
                         "Choose from: linear, tree, kernel.")
    else:
        estimator.type = type

    return estimator


@typechecked
def ATOMLoader(filename: str,
               X: Optional[X_TYPES] = None,
               y: Y_TYPES = -1,
               n_rows: Union[int, float] = 1,
               transform_data: bool = True,
               verbose: Optional[int] = None):
    """Load a class instance from a pickle file.

    If the file is a `training` instance that was saved using `save_data=False`,
    you can load new data into it. If the file is an `atom` instance, you can also
    apply all data transformations in the pipeline to the provided data.

    Parameters
    ----------
    filename: str
        Name of the pickle file to load.

    X: dict, list, tuple, np.array, pd.DataFrame or None, optional (default=None)
        Data containing the features, with shape=(n_samples, n_features). Only
        use this parameter for `training` instances saved using `save_data=False`.
        See the save method in basetransformer.py.

    y: int, str, dict or array-like, optional (default=-1)
        - If int: Index of the target column in X.
        - If str: Name of the target column in X.
        - Else: Target column with shape=(n_samples,).

        This parameter is ignored if X=None.

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the data to use.
        - If >1: Number of rows to use.

        This parameter is ignored if X=None.

    transform_data: bool, optional (default=True)
        Whether to transform the provided data through all the steps in the
        instance's pipeline. This parameter is ignored if the loaded file is
        not an `atom` instance.

    verbose: int or None, optional (default=None)
        Verbosity level of the transformations applied on the new data. If None,
        use the verbosity from the loaded instance. This parameter is ignored if
        the loaded file is not an `atom` instance.

    """
    # Check verbose parameter
    if verbose and (verbose < 0 or verbose > 2):
        raise ValueError("Invalid value for the verbose parameter." +
                         f"Value should be between 0 and 2, got {verbose}.")

    with open(filename, 'rb') as f:
        cls_ = pickle.load(f)

    if X is not None:
        if not hasattr(cls_, '_data'):
            raise TypeError("X is provided but the class is not an ATOM nor " +
                            f"training instance, got {cls_.__class__.__name__}.")

        elif cls_._data is not None:
            raise ValueError("The loaded {} instance already contains data!"
                             .format(cls_.__class__.__name__))

        # Prepare the provided data
        X, y = BaseTransformer._prepare_input(X, y)
        cls_._data = X if y is None else merge(X, y)

        # Get number of rows and shuffle the dataset
        kwargs = {'frac': n_rows} if n_rows <= 1 else {'n': int(n_rows)}
        cls_._data = cls_._data.sample(random_state=cls_.random_state, **kwargs)

        if hasattr(cls_, 'pipeline') and transform_data:
            # Transform the data through all transformers in the pipeline
            for estimator in [i for i in cls_.pipeline if hasattr(i, 'transform')]:
                if verbose is not None:
                    vb = estimator.get_params()['verbose']  # Save original verbosity
                    estimator.set_params(verbose=verbose)

                # Some transformations are only applied on the training set
                if estimator.__class__.__name__ in ['Outliers', 'Balancer']:
                    X, y = estimator.transform(cls_.X_train, cls_.y_train)
                    cls_._data = pd.concat([merge(X, y), cls_.test])
                else:
                    X = estimator.transform(cls_.X, cls_.y)

                    # Data changes depending if the estimator returned X or X, y
                    cls_._data = merge(*X) if isinstance(X, tuple) else merge(X, cls_.y)

                cls_._data.reset_index(drop=True, inplace=True)
                if verbose is not None:
                    estimator.verbose = vb  # Reset the original verbosity

        if getattr(cls_, 'trainer', None):
            cls_.trainer._data = cls_._data

    cls_.log(f"{cls_.__class__.__name__} loaded successfully!", 1)

    return cls_


# Classes =================================================================== >>

class ATOMClassifier(BaseTransformer, ATOM):
    """ATOM class for classification tasks.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing the features and target. Allowed formats are:
            - X, y
            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        X, train, test: dict, list, tuple, np.array or pd.DataFrame
            Feature set with shape=(n_features, n_samples). If
            no y is provided, the last column is used as target.

        y: int, str, dict or array-like
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the dataset to use.
        - If >1: Number of rows to use (only if input is X, y).

    test_size: int, float, optional (default=0.2)
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        Is ignored if the train and test set are provided.

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

        Note that ATOM can't manage warnings that go directly from C/C++ code
        to the stdout/stderr.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. 'auto' for default name.
        - If class: python 'Logger' object.

        Note that warnings will not be saved to the logger in any case.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` instance used by `numpy.random`.

    """

    @typechecked
    def __init__(self,
                 *arrays,
                 n_rows: Union[int, float] = 1,
                 test_size: float = 0.2,
                 logger: Optional[Union[str, callable]] = None,
                 n_jobs: int = 1,
                 warnings: Union[bool, str] = True,
                 verbose: int = 0,
                 random_state: Optional[int] = None):
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         warnings=warnings,
                         logger=logger,
                         random_state=random_state)

        self.goal = 'classification'
        ATOM.__init__(self, *arrays, n_rows=n_rows, test_size=test_size)


class ATOMRegressor(BaseTransformer, ATOM):
    """ATOM class for regression tasks.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing the features and target. Allowed formats are:
            - X, y
            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        X, train, test: dict, list, tuple, np.array or pd.DataFrame
            Feature set with shape=(n_features, n_samples). If no
            y is provided, the last column is used as target.

        y: int, str, dict or array-like
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

    n_rows: int or float, optional (default=1)
        - If <=1: Fraction of the dataset to use.
        - If >1: Number of rows to use (only if input is X, y).

    test_size: int, float, optional (default=0.2)
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        Is ignored if the train and test set are provided.

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

        Note that ATOM can't manage warnings that go directly from C/C++ code
        to the stdout/stderr.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. 'auto' for default name.
        - If class: python 'Logger' object.

        Note that warnings will not be saved to the logger in any case.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` instance used by `numpy.random`.

    """

    @typechecked
    def __init__(self,
                 *arrays,
                 n_rows: Union[int, float] = 1,
                 test_size: float = 0.2,
                 n_jobs: int = 1,
                 warnings: Union[bool, str] = True,
                 verbose: int = 0,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         warnings=warnings,
                         logger=logger,
                         random_state=random_state)

        self.goal = 'regression'
        ATOM.__init__(self, *arrays, n_rows=n_rows, test_size=test_size)
